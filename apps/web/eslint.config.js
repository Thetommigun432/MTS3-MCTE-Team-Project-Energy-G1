import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import react from "eslint-plugin-react"; // New dependency
import tailwind from "eslint-plugin-tailwindcss"; // New dependency
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist"] },
  {
    extends: [
      js.configs.recommended,
      ...tseslint.configs.recommended,
      ...tailwind.configs["flat/recommended"], // Adds Tailwind rules
    ],
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
      parserOptions: {
        ecmaFeatures: {
          jsx: true, // Enable JSX parsing
        },
      },
    },
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
      react, // Register React plugin
    },
    // Define React version explicitly for React 19 detection
    settings: {
      react: {
        version: "19.0",
      },
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      // Load recommended React runtime rules (key props, etc.)
      ...react.configs.flat.recommended.rules,

      // React 19 doesn't need React imported in JSX files anymore
      "react/react-in-jsx-scope": "off",

      // Tailwind v4 Customization
      // This ensures custom classes (like "text-brand-primary") don't error if defined in CSS
      "tailwindcss/no-custom-classname": "off",

      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true },
      ],
      "@typescript-eslint/no-unused-vars": [
        "off", // Delegating strictness to TS compiler usually suffices
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
        },
      ],
    },
  },
  // Keep your existing override for UI components
  {
    files: ["src/components/ui/**/*.{ts,tsx}", "src/contexts/**/*.{ts,tsx}"],
    rules: {
      "react-refresh/only-export-components": "off",
    },
  },
);
