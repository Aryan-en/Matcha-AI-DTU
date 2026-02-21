import { ExpoConfig, ConfigContext } from "expo/config";

export default ({ config }: ConfigContext): ExpoConfig => ({
  ...config,
  name: "Matcha AI",
  slug: "matcha-ai",
  version: "1.0.0",
  orientation: "portrait",
  icon: "./assets/icon.png",
  userInterfaceStyle: "dark",
  splash: {
    image: "./assets/splash.png",
    resizeMode: "contain",
    backgroundColor: "#07080F",
  },
  android: {
    adaptiveIcon: {
      foregroundImage: "./assets/adaptive-icon.png",
      backgroundColor: "#07080F",
    },
    package: "com.matchaai.dtu",
  },
  plugins: ["expo-router"],
  scheme: "matcha-ai",
  experiments: {
    typedRoutes: true,
  },
});
