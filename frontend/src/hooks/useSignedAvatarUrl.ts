import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";

/**
 * Hook to generate and manage signed URLs for private avatar storage
 * Signed URLs expire after 1 hour and are automatically refreshed
 */
export function useSignedAvatarUrl(
  avatarPath: string | null | undefined,
): string | null {
  const [signedUrl, setSignedUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!avatarPath) {
      setSignedUrl(null);
      return;
    }

    // Extract the file path from the full URL if it's a public URL
    // Format: https://...supabase.co/storage/v1/object/public/avatars/userId/timestamp.ext
    let filePath = avatarPath;

    // Normalize legacy values that were stored with bucket prefix
    if (filePath.startsWith("avatars/")) {
      filePath = filePath.replace(/^avatars\//, "");
    }

    if (avatarPath.includes("/storage/v1/object/")) {
      const match = avatarPath.match(/\/avatars\/(.+)$/);
      if (match) {
        filePath = match[1];
      }
    }

    const getSignedUrl = async () => {
      try {
        const { data, error } = await supabase.storage
          .from("avatars")
          .createSignedUrl(filePath, 3600); // 1 hour expiry

        if (error) {
          console.error("Failed to get signed URL:", error);
          setSignedUrl(null);
          return;
        }

        setSignedUrl(data.signedUrl);
      } catch (err) {
        console.error("Error getting signed avatar URL:", err);
        setSignedUrl(null);
      }
    };

    getSignedUrl();

    // Refresh the signed URL every 50 minutes (before 1 hour expiry)
    const refreshInterval = setInterval(getSignedUrl, 50 * 60 * 1000);

    return () => clearInterval(refreshInterval);
  }, [avatarPath]);

  return signedUrl;
}
