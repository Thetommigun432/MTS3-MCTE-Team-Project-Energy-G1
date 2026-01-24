type IdleHandle = number;

// schedule a callback during idle time with a timeout fallback
export function scheduleIdle(fn: () => void, fallbackDelay = 1): IdleHandle {
  if (typeof window === "undefined") {
    // Non-browser environment - return a dummy handle
    return 0 as IdleHandle;
  }
  const win = window as Window & typeof globalThis & {
    requestIdleCallback?: (cb: IdleRequestCallback) => IdleHandle
  };
  if (win.requestIdleCallback) {
    return win.requestIdleCallback(fn);
  }
  return win.setTimeout(fn, fallbackDelay);
}

// cancel a scheduled idle callback or timeout
scheduleIdle.cancel = function cancel(handle: IdleHandle) {
  if (typeof window === "undefined") {
    return;
  }
  const win = window as Window & typeof globalThis & {
    cancelIdleCallback?: (handle: IdleHandle) => void
  };
  if (win.cancelIdleCallback) {
    win.cancelIdleCallback(handle);
  } else {
    win.clearTimeout(handle);
  }
};
