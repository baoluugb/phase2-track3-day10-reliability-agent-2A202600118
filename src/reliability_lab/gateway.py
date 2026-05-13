from __future__ import annotations

from dataclasses import dataclass
import time

from reliability_lab.cache import ResponseCache, SharedRedisCache
from reliability_lab.circuit_breaker import CircuitBreaker, CircuitOpenError
from reliability_lab.providers import FakeLLMProvider, ProviderError, ProviderResponse


@dataclass(slots=True)
class GatewayResponse:
    text: str
    route: str
    provider: str | None
    cache_hit: bool
    latency_ms: float
    estimated_cost: float
    error: str | None = None


class ReliabilityGateway:
    """Routes requests through cache, circuit breakers, and fallback providers."""

    def __init__(
        self,
        providers: list[FakeLLMProvider],
        breakers: dict[str, CircuitBreaker],
        cache: ResponseCache | SharedRedisCache | None = None,
        cost_budget: float | None = None,
    ):
        self.providers = providers
        self.breakers = breakers
        self.cache = cache
        self.cost_budget = cost_budget
        self.total_cost = 0.0
        self._primary_name = providers[0].name if providers else None
        self._cheapest_cost = min(
            (p.cost_per_1k_tokens for p in providers), default=0.0)

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        if isinstance(exc, CircuitOpenError):
            return "circuit_open"
        if isinstance(exc, ProviderError):
            return "provider_error"
        return "unknown_error"

    @staticmethod
    def _format_route(
        kind: str,
        provider_name: str | None,
        last_error_kind: str | None,
        budget_reason: str | None,
    ) -> str:
        parts: list[str] = [kind]
        if provider_name:
            parts.append(provider_name)
        if budget_reason:
            parts.append(budget_reason)
        if kind == "fallback" and last_error_kind:
            parts.append(f"after_{last_error_kind}")
        if kind == "primary" and provider_name and not budget_reason:
            parts.append("direct")
        return ":".join(parts)

    def complete(self, prompt: str) -> GatewayResponse:
        """Return a reliable response or a static fallback.

        TODO(student): Improve route reasons, cache safety checks, and error handling.
        TODO(student): Add cost budget check — if cumulative cost exceeds a threshold,
        skip expensive providers and route to cache or cheaper fallback.
        """
        start = time.perf_counter()
        last_error: str | None = None
        last_error_kind: str | None = None

        if self.cache is not None:
            try:
                cached, score = self.cache.get(prompt)
            except Exception as exc:
                last_error = str(exc)
                last_error_kind = self._classify_error(exc)
                cached = None
                score = 0.0
            if cached is not None:
                latency_ms = (time.perf_counter() - start) * 1000
                return GatewayResponse(cached, f"cache_hit:{score:.2f}", None, True, latency_ms, 0.0)

        providers = self.providers
        budget_reason: str | None = None
        if self.cost_budget is not None and self.total_cost >= self.cost_budget:
            providers = [
                p for p in self.providers if p.cost_per_1k_tokens == self._cheapest_cost]
            if not providers:
                providers = self.providers
            budget_reason = "budget_exceeded"

        for provider in providers:
            breaker = self.breakers[provider.name]
            try:
                response = breaker.call(provider.complete, prompt)
            except (ProviderError, CircuitOpenError) as exc:
                last_error = str(exc)
                last_error_kind = self._classify_error(exc)
                continue
            if self.cache is not None:
                try:
                    self.cache.set(prompt, response.text, {
                                   "provider": provider.name})
                except Exception:
                    pass
            self.total_cost += response.estimated_cost
            route_kind = "primary" if provider.name == self._primary_name else "fallback"
            route = self._format_route(
                route_kind, provider.name, last_error_kind, budget_reason)
            latency_ms = (time.perf_counter() - start) * 1000
            return GatewayResponse(
                text=response.text,
                route=route,
                provider=provider.name,
                cache_hit=False,
                latency_ms=latency_ms,
                estimated_cost=response.estimated_cost,
            )

        latency_ms = (time.perf_counter() - start) * 1000
        return GatewayResponse(
            text="The service is temporarily degraded. Please try again soon.",
            route="static_fallback",
            provider=None,
            cache_hit=False,
            latency_ms=latency_ms,
            estimated_cost=0.0,
            error=last_error,
        )
