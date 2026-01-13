import { createContext, useContext, useEffect, useState, ReactNode } from 'react';

interface AppearanceSettings {
  highContrast: boolean;
  compactMode: boolean;
  showAnimations: boolean;
}

interface AppearanceContextType {
  settings: AppearanceSettings;
  setHighContrast: (enabled: boolean) => void;
  setCompactMode: (enabled: boolean) => void;
  setShowAnimations: (enabled: boolean) => void;
}

const AppearanceContext = createContext<AppearanceContextType | undefined>(undefined);

const STORAGE_KEY = 'nilm-appearance';

const defaultSettings: AppearanceSettings = {
  highContrast: false,
  compactMode: false,
  showAnimations: true,
};

export function AppearanceProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<AppearanceSettings>(() => {
    if (typeof window === 'undefined') return defaultSettings;
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        return { ...defaultSettings, ...JSON.parse(stored) };
      } catch {
        return defaultSettings;
      }
    }
    return defaultSettings;
  });

  // Apply high contrast class to document
  useEffect(() => {
    const root = document.documentElement;
    if (settings.highContrast) {
      root.classList.add('high-contrast');
    } else {
      root.classList.remove('high-contrast');
    }
  }, [settings.highContrast]);

  // Apply compact mode class
  useEffect(() => {
    const root = document.documentElement;
    if (settings.compactMode) {
      root.classList.add('compact-mode');
    } else {
      root.classList.remove('compact-mode');
    }
  }, [settings.compactMode]);

  // Apply animations class
  useEffect(() => {
    const root = document.documentElement;
    if (!settings.showAnimations) {
      root.classList.add('reduce-motion');
    } else {
      root.classList.remove('reduce-motion');
    }
  }, [settings.showAnimations]);

  // Persist settings
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  }, [settings]);

  const setHighContrast = (enabled: boolean) => {
    setSettings(prev => ({ ...prev, highContrast: enabled }));
  };

  const setCompactMode = (enabled: boolean) => {
    setSettings(prev => ({ ...prev, compactMode: enabled }));
  };

  const setShowAnimations = (enabled: boolean) => {
    setSettings(prev => ({ ...prev, showAnimations: enabled }));
  };

  return (
    <AppearanceContext.Provider value={{
      settings,
      setHighContrast,
      setCompactMode,
      setShowAnimations,
    }}>
      {children}
    </AppearanceContext.Provider>
  );
}

export function useAppearance() {
  const context = useContext(AppearanceContext);
  if (!context) {
    throw new Error('useAppearance must be used within AppearanceProvider');
  }
  return context;
}
