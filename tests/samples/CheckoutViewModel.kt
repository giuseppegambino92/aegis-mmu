package com.ledgerline.checkout

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch

data class CheckoutState(val total: Long = 0, val error: String? = null)

class CheckoutViewModel(private val repo: CheckoutRepository) : ViewModel() {
    private val _state = MutableStateFlow(CheckoutState())

    fun submit(cartId: String) {
        viewModelScope.launch {
            val result = repo.checkout(cartId)
            if (result.isSuccess) {
                _state.value = CheckoutState(total = result.getOrThrow().total)
            } else {
                _state.value = CheckoutState(error = result.exceptionOrNull()?.message)
            }
        }
    }

    private fun applyDiscount(total: Long, percent: Int): Long {
        require(percent in 0..100) { "bad percent" }
        return total - total * percent / 100
    }

    companion object {
        const val MAX_RETRIES = 3
    }
}
