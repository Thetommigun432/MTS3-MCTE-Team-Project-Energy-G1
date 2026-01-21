import { useState, useCallback, useRef, useEffect } from "react";

interface RateLimitState {
  isLimited: boolean;
  remainingTime: number;
  attempt: () => boolean;
  reset: () => void;
}

/**
 * Hook for client-side rate limiting
 * @param maxAttempts - Maximum attempts allowed in the time window
 * @param windowMs - Time window in milliseconds
 * @param cooldownMs - Cooldown period after limit is hit
 */
export function useRateLimit(
  maxAttempts: number = 3,
  windowMs: number = 60000, // 1 minute
  cooldownMs: number = 60000, // 1 minute cooldown
): RateLimitState {
  const [isLimited, setIsLimited] = useState(false);
  const [remainingTime, setRemainingTime] = useState(0);
  const attemptsRef = useRef<number[]>([]);
  const cooldownEndRef = useRef<number>(0);

  // Check and update remaining time
  useEffect(() => {
    if (!isLimited) return;

    const interval = setInterval(() => {
      const now = Date.now();
      const remaining = Math.max(
        0,
        Math.ceil((cooldownEndRef.current - now) / 1000),
      );

      if (remaining === 0) {
        setIsLimited(false);
        setRemainingTime(0);
        attemptsRef.current = [];
      } else {
        setRemainingTime(remaining);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [isLimited]);

  const attempt = useCallback((): boolean => {
    const now = Date.now();

    // If in cooldown, reject
    if (now < cooldownEndRef.current) {
      return false;
    }

    // Clean old attempts outside the window
    attemptsRef.current = attemptsRef.current.filter(
      (timestamp) => now - timestamp < windowMs,
    );

    // Check if we've exceeded the limit
    if (attemptsRef.current.length >= maxAttempts) {
      cooldownEndRef.current = now + cooldownMs;
      setIsLimited(true);
      setRemainingTime(Math.ceil(cooldownMs / 1000));
      return false;
    }

    // Record this attempt
    attemptsRef.current.push(now);
    return true;
  }, [maxAttempts, windowMs, cooldownMs]);

  const reset = useCallback(() => {
    attemptsRef.current = [];
    cooldownEndRef.current = 0;
    setIsLimited(false);
    setRemainingTime(0);
  }, []);

  return { isLimited, remainingTime, attempt, reset };
}
