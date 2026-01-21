type IdleHandle = number;

// schedule a callback during idle time with a timeout fallback
export function scheduleIdle(fn: () => void, fallbackDelay = 1): IdleHandle {
  if (typeof window !== "undefined" && "requestIdleCallback" in window) {
    // requestIdleCallback is not universally available; cast to access when present
    return (window as Window & { requestIdleCallback?: (cb: IdleRequestCallback) => IdleHandle }).requestIdleCallback?.(fn) ?? window.setTimeout(fn, fallbackDelay);
  }
  return window.setTimeout(fn, fallbackDelay);
}

// cancel a scheduled idle callback or timeout
scheduleIdle.cancel = function cancel(handle: IdleHandle) {
  if (typeof window !== "undefined" && "cancelIdleCallback" in window) {
    (window as Window & { cancelIdleCallback?: (handle: IdleHandle) => void }).cancelIdleCallback?.(handle);
  } else {
    clearTimeout(handle);
  }
};
