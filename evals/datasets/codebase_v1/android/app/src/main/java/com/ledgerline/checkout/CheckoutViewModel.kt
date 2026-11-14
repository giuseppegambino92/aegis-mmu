package com.ledgerline.checkout

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
import java.util.UUID

sealed interface CheckoutStep {
    data object Loading : CheckoutStep
    data object Shipping : CheckoutStep
    data object Payment : CheckoutStep
    data class Confirmed(val orderId: String) : CheckoutStep
    data class Failed(val message: String) : CheckoutStep
}

data class CheckoutUiState(
    val step: CheckoutStep = CheckoutStep.Loading,
    val shippingOptions: List<ShippingOption> = emptyList(),
    val selectedShippingId: String? = null,
    val total: Money? = null,
    val isSubmitting: Boolean = false,
)

sealed interface PaymentSheetResult {
    data class Completed(val paymentIntentId: String) : PaymentSheetResult
    data object Canceled : PaymentSheetResult
    data class Failed(val reason: String) : PaymentSheetResult
}

class CheckoutViewModel(
    private val repository: CheckoutRepository,
    savedState: SavedStateHandle,
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

    init {
        viewModelScope.launch {
            try {
                val options = repository.shippingOptions(cartId)
                val preselected = options.firstOrNull { it.isDefault } ?: options.firstOrNull()
                _state.update {
                    it.copy(step = CheckoutStep.Shipping, shippingOptions = options, selectedShippingId = preselected?.id)
                }
                preselected?.let { requote(it.id) }
            } catch (error: CancellationException) {
                throw error
            } catch (error: Exception) {
                _state.update { it.copy(step = CheckoutStep.Failed(error.userMessage())) }
            }
        }
    }

    fun selectShipping(optionId: String) {
        _state.update { it.copy(selectedShippingId = optionId, total = null) }
        requote(optionId)
    }

    private fun requote(optionId: String) {
        quoteJob?.cancel()
        quoteJob = viewModelScope.launch {
            delay(QUOTE_DEBOUNCE_MS)
            val quote = repository.quote(cartId, optionId)
            _state.update { it.copy(total = quote.total) }
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
                val intent = repository.createPaymentIntent(cartId, shippingId, idempotencyKey)
                _state.update { it.copy(step = CheckoutStep.Payment) }
                _paymentSheet.emit(intent)
            } catch (error: CancellationException) {
                throw error
            } catch (error: Exception) {
                _state.update { it.copy(step = CheckoutStep.Failed(error.userMessage())) }
            } finally {
                _state.update { it.copy(isSubmitting = false) }
            }
        }
    }

    fun onPaymentSheetResult(result: PaymentSheetResult) {
        when (result) {
            is PaymentSheetResult.Completed -> confirmOrder(result.paymentIntentId)
            is PaymentSheetResult.Canceled -> _state.update { it.copy(step = CheckoutStep.Shipping) }
            is PaymentSheetResult.Failed -> _state.update { it.copy(step = CheckoutStep.Failed(result.reason)) }
        }
    }

    private fun confirmOrder(paymentIntentId: String) {
        viewModelScope.launch {
            try {
                val orderId = repository.confirm(cartId, paymentIntentId)
                _state.update { it.copy(step = CheckoutStep.Confirmed(orderId)) }
            } catch (error: CancellationException) {
                throw error
            } catch (error: Exception) {
                // The payment may still have succeeded; the order is reconciled by webhook.
                _state.update { it.copy(step = CheckoutStep.Failed(PENDING_CONFIRMATION_MESSAGE)) }
            }
        }
    }

    private fun Throwable.userMessage(): String =
        message?.takeIf { it.isNotBlank() } ?: "Something went wrong. Please try again."

    companion object {
        private const val KEY_IDEMPOTENCY = "checkout_idempotency_key"
        private const val QUOTE_DEBOUNCE_MS = 250L
        private const val PENDING_CONFIRMATION_MESSAGE = "We're confirming your payment. Check your orders shortly."
    }
}
