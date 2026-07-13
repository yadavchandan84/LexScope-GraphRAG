import time

import google.generativeai as genai
from google.api_core.exceptions import (
    PermissionDenied,
    ResourceExhausted,
    ServiceUnavailable,
    Unauthenticated,
)


class RateLimitError(Exception):
    pass


class AuthError(Exception):
    pass


class ServerError(Exception):
    pass


class AllPairsExhaustedError(Exception):
    pass


class GeminiRotator:
    def __init__(self, api_keys: list[str], models: list[str]):
        self.pairs = [(k, m) for m in models for k in api_keys]
        self.cooldowns: dict[tuple[str, str], float] = {}
        self._idx = 0

    def _next_available_pair(self) -> tuple[str, str] | None:
        now = time.time()
        for _ in range(len(self.pairs)):
            pair = self.pairs[self._idx % len(self.pairs)]
            self._idx += 1
            unblock_at = self.cooldowns.get(pair)
            if unblock_at is None or unblock_at <= now:
                return pair
        return None

    def _call(
        self,
        key: str,
        model: str,
        prompt: str,
        system_instruction: str,
        response_schema: dict,
        temperature: float,
    ) -> str:
        genai.configure(api_key=key)
        gen_model = genai.GenerativeModel(model_name=model, system_instruction=system_instruction)
        try:
            response = gen_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
            return response.text
        except ResourceExhausted as e:
            raise RateLimitError(str(e)) from e
        except (PermissionDenied, Unauthenticated) as e:
            raise AuthError(str(e)) from e
        except ServiceUnavailable as e:
            raise ServerError(str(e)) from e

    def generate(
        self,
        prompt: str,
        system_instruction: str,
        response_schema: dict,
        temperature: float = 0.1,
    ) -> str:
        attempts = 0
        while attempts < len(self.pairs):
            pair = self._next_available_pair()
            if pair is None:
                raise AllPairsExhaustedError("All (key, model) pairs are in cooldown.")
            key, model = pair
            try:
                return self._call(key, model, prompt, system_instruction, response_schema, temperature)
            except RateLimitError:
                self.cooldowns[pair] = time.time() + 60
                attempts += 1
                continue
            except AuthError:
                # Invalid/unauthorized key — park it for a long time and rotate on.
                self.cooldowns[pair] = time.time() + 86400
                attempts += 1
                continue
            except ServerError:
                time.sleep(1)
                try:
                    return self._call(key, model, prompt, system_instruction, response_schema, temperature)
                except (RateLimitError, ServerError):
                    attempts += 1
                    continue
        raise AllPairsExhaustedError("All (key, model) pairs exhausted after retries.")
