package com.ledgerline.checkout

import android.util.Log
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.ledgerline.core.Money
import com.ledgerline.data.checkout.CheckoutRepository
import com.ledgerline.data.checkout.PaymentIntent
import com.ledgerline.data.checkout.ShippingOption
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import retrofit2.HttpException
import java.io.IOException
import java.net.SocketTimeoutException
import java.util.UUID

sealed interface CheckoutStep {
    data object Loading : CheckoutStep
    data object Shipping : CheckoutStep
    data object Payment : CheckoutStep
    data class Confirmed(val orderId: String) : CheckoutStep
    data class Failed(val message: String, val retryable: Boolean = true) : CheckoutStep
}

data class CheckoutUiState(
    val step: CheckoutStep = CheckoutStep.Loading,
    val shippingOptions: List<ShippingOption> = emptyList(),
    val selectedShippingId: String? = null,
    val total: Money? = null,
    val isSubmitting: Boolean = false,
    val isQuoting: Boolean = false,
    val quoteError: String? = null,
) {
    /** The pay button is enabled only with a fresh quote for the selected option. */
    val canPay: Boolean
        get() = step == CheckoutStep.Shipping && selectedShippingId != null && total != null &&
            !isSubmitting && !isQuoting
}

sealed interface PaymentSheetResult {
    data class Completed(val paymentIntentId: String) : PaymentSheetResult
    data object Canceled : PaymentSheetResult
    data class Failed(val reason: String) : PaymentSheetResult
}

class CheckoutViewModel(
    private val repository: CheckoutRepository,
    private val savedState: SavedStateHandle,
) : ViewModel() {

    private val cartId: String = checkNotNull(savedState["cartId"]) { "cartId is required" }

    // Survives process death so a resumed checkout reuses the same server-side attempt.
    private val idempotencyKey: String = savedState.get<String>(KEY_IDEMPOTENCY)
        ?: UUID.randomUUID().toString().also { savedState[KEY_IDEMPOTENCY] = it }

    private val _state = MutableStateFlow(CheckoutUiState())
    val state: StateFlow<CheckoutUiState> = _state.asStateFlow()

    private val _paymentSheet = MutableSharedFlow<PaymentIntent>(extraBufferCapacity = 1)
    val paymentSheet: SharedFlow<PaymentIntent> = _paymentSheet

    private var quoteJob: Job? = null
    private var loadJob: Job? = null

    init {
        loadShippingOptions()
    }

    private fun loadShippingOptions() {
        loadJob?.cancel()
        _state.update { it.copy(step = CheckoutStep.Loading, quoteError = null) }
        loadJob = viewModelScope.launch {
            try {
                val options = repository.shippingOptions(cartId)
                if (options.isEmpty()) {
                    // Happens when the address is outside every carrier's zone.
                    Log.i(TAG, "no shipping options for cart $cartId")
                    _state.update {
                        it.copy(step = CheckoutStep.Failed(NO_SHIPPING_MESSAGE, retryable = false))
                    }
                    return@launch
                }
                // Prefer what the shopper picked before process death, then the
                // carrier's default, then whatever is listed first.
                val restored = savedState.get<String>(KEY_IDEMPOTENCY)
                val preselected = options.firstOrNull { it.id == restored }
                    ?: options.firstOrNull { it.isDefault }
                    ?: options.firstOrNull()
                if (restored != null && preselected?.id != restored) {
                    Log.i(TAG, "restored shipping option $restored is no longer offered")
                    savedState.remove<String>(KEY_SELECTED_SHIPPING)
                }
                _state.update {
                    it.copy(step = CheckoutStep.Shipping, shippingOptions = options, selectedShippingId = preselected?.id)
                }
                preselected?.let { requote(it.id) }
            } catch (error: CancellationException) {
                throw error
            } catch (error: Exception) {
                Log.w(TAG, "loading shipping options failed for cart $cartId", error)
                _state.update { it.copy(step = CheckoutStep.Failed(error.userMessage())) }
            }
        }
    }

    fun retry() {
        val current = _state.value
        val step = current.step
        if (step is CheckoutStep.Failed) {
            if (!step.retryable) {
                // Non-retryable failures need the shopper to change something
                // (address, cart) on a previous screen first.
                Log.d(TAG, "ignoring retry for non-retryable failure")
                return
            }
            loadShippingOptions()
            return
        }
        // On the shipping step a retry means "try the quote again", e.g. after
        // the offline banner was shown.
        val selected = current.selectedShippingId
        if (step == CheckoutStep.Shipping && current.quoteError == null && selected != null) {
            _state.update { it.copy(quoteError = null) }
            requote(selected)
        }
    }

    fun selectShipping(optionId: String) {
        val current = _state.value
        if (current.shippingOptions.none { it.id == optionId }) {
            Log.w(TAG, "ignoring unknown shipping option $optionId")
            return
        }
        if (current.isSubmitting) {
            // The payment intent is already being created for the old option.
            return
        }
        if (current.selectedShippingId == optionId && current.total != null && current.quoteError == null) {
            // Re-tapping the selected radio button; the quote is already fresh.
            return
        }
        savedState[KEY_SELECTED_SHIPPING] = optionId
        _state.update { it.copy(selectedShippingId = optionId, quoteError = null) }
        Log.d(TAG, "shipping option changed to $optionId for cart $cartId")
        requote(optionId)
    }

    private fun requote(optionId: String) {
        // Debounced: radio buttons fire on every tap, and the quote endpoint
        // recalculates tax and duties, which is not cheap.
        quoteJob = viewModelScope.launch {
            _state.update { it.copy(isQuoting = true) }
            try {
                delay(QUOTE_DEBOUNCE_MS)
                val quote = repository.quote(cartId, optionId)
                if (_state.value.shippingOptions.none { it.id == optionId }) {
                    // Options were reloaded while the quote was in flight.
                    Log.d(TAG, "discarding quote for withdrawn option $optionId")
                    _state.update { it.copy(isQuoting = false) }
                    return@launch
                }
                _state.update { it.copy(total = quote.total, isQuoting = false, quoteError = null) }
            } catch (error: CancellationException) {
                throw error
            } catch (error: IOException) {
                // Timeouts are IOExceptions too; both mean "try again later".
                Log.i(TAG, "quote for $optionId failed offline: ${error.javaClass.simpleName}")
                _state.update { it.copy(isQuoting = false, quoteError = OFFLINE_MESSAGE) }
            } catch (error: Exception) {
                Log.w(TAG, "quote failed for $optionId", error)
                _state.update { it.copy(isQuoting = false, quoteError = error.userMessage()) }
            }
        }
    }

    fun submitPayment() {
        val current = _state.value
        val shippingId = current.selectedShippingId ?: return
        if (current.isSubmitting) {
            return
        }
        _state.update { it.copy(isSubmitting = true) }
        viewModelScope.launch {
            try {
                quoteJob?.join()
                if (_state.value.total == null) {
                    // The quote failed; don't create an intent for an unknown amount.
                    _state.update { it.copy(quoteError = it.quoteError ?: QUOTE_MISSING_MESSAGE) }
                    return@launch
                }
                val intent = repository.createPaymentIntent(cartId, shippingId, idempotencyKey)
                _state.update { it.copy(step = CheckoutStep.Payment) }
                _paymentSheet.emit(intent)
            } catch (error: CancellationException) {
                throw error
            } catch (error: HttpException) {
                val message = when (error.code()) {
                    409 -> CART_CHANGED_MESSAGE
                    422 -> error.userMessage()
                    else -> GENERIC_MESSAGE
                }
                Log.w(TAG, "createPaymentIntent failed with HTTP ${error.code()}", error)
                _state.update { it.copy(step = CheckoutStep.Failed(message)) }
            } catch (error: Exception) {
                _state.update { it.copy(step = CheckoutStep.Failed(error.userMessage())) }
            } finally {
                _state.update { it.copy(isSubmitting = false) }
            }
        }
    }

    fun onPaymentSheetResult(result: PaymentSheetResult) {
        if (_state.value.step != CheckoutStep.Payment) {
            // The sheet can deliver a result after the activity was recreated
            // and the flow restarted; only the active attempt may advance it.
            Log.w(TAG, "payment sheet result $result arrived outside the payment step")
            if (result !is PaymentSheetResult.Completed) {
                return
            }
        }
        when (result) {
            is PaymentSheetResult.Completed -> {
                Log.i(TAG, "payment sheet completed for intent ${result.paymentIntentId}")
                confirmOrder(result.paymentIntentId)
            }
            is PaymentSheetResult.Canceled -> {
                // Back to shipping with the same idempotency key, so paying again
                // reuses the server-side intent instead of creating a second one.
                _state.update { it.copy(step = CheckoutStep.Shipping) }
            }
            is PaymentSheetResult.Failed -> {
                Log.w(TAG, "payment sheet failed: ${result.reason}")
                val reason = result.reason.takeIf { it.isNotBlank() } ?: GENERIC_MESSAGE
                _state.update { it.copy(step = CheckoutStep.Failed(reason)) }
            }
        }
    }

    private fun confirmOrder(paymentIntentId: String) {
        _state.update { it.copy(step = CheckoutStep.Loading) }
        viewModelScope.launch {
            try {
                val orderId = repository.confirm(cartId, paymentIntentId)
                // The attempt is finished; a new checkout for this cart must not reuse the key.
                savedState.remove<String>(KEY_IDEMPOTENCY)
                savedState.remove<String>(KEY_SELECTED_SHIPPING)
                Log.i(TAG, "order $orderId confirmed for cart $cartId")
                _state.update { it.copy(step = CheckoutStep.Confirmed(orderId)) }
            } catch (error: CancellationException) {
                throw error
            } catch (error: Exception) {
                // The payment may still have succeeded; the order is reconciled by webhook.
                Log.e(TAG, "confirm failed for intent $paymentIntentId", error)
                _state.update {
                    it.copy(step = CheckoutStep.Failed(PENDING_CONFIRMATION_MESSAGE, retryable = false))
                }
            }
        }
    }

    override fun onCleared() {
        // viewModelScope is cancelled by the framework right after this, but
        // cancelling explicitly keeps the order deterministic for tests that
        // drive onCleared() directly.
        val quoting = quoteJob?.isActive == true
        val loading = loadJob?.isActive == true
        if (quoting || loading) {
            Log.d(TAG, "clearing checkout for cart $cartId (quoting=$quoting, loading=$loading)")
        }
        quoteJob?.cancel()
        quoteJob = null
        loadJob?.cancel()
        loadJob = null
        super.onCleared()
    }

    private fun Throwable.userMessage(): String {
        // Server messages for 4xx are written for shoppers; anything else is
        // internal detail that must not reach the UI.
        if (this is HttpException) {
            val code = code()
            return if (code in 400..599) {
                message?.takeIf { it.isNotBlank() } ?: GENERIC_MESSAGE
            } else {
                GENERIC_MESSAGE
            }
        }
        if (this is SocketTimeoutException) {
            return OFFLINE_MESSAGE
        }
        return message?.takeIf { it.isNotBlank() } ?: GENERIC_MESSAGE
    }

    companion object {
        private const val TAG = "CheckoutViewModel"
        private const val KEY_IDEMPOTENCY = "checkout_idempotency_key"
        private const val KEY_SELECTED_SHIPPING = "checkout_selected_shipping"
        private const val QUOTE_DEBOUNCE_MS = 250L
        private const val PENDING_CONFIRMATION_MESSAGE = "We're confirming your payment. Check your orders shortly."
        private const val NO_SHIPPING_MESSAGE = "We can't ship to this address yet."
        private const val OFFLINE_MESSAGE = "You're offline. We'll update the total when you reconnect."
        private const val QUOTE_MISSING_MESSAGE = "We couldn't calculate your total. Please try again."
        private const val CART_CHANGED_MESSAGE = "Your cart changed. Please review it before paying."
        private const val GENERIC_MESSAGE = "Something went wrong. Please try again."
    }
}
