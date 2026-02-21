// Match detail — events, highlights, video player, intensity chart
// TODO: wire up useMatchDetail hook
import { useLocalSearchParams } from "expo-router";
import { View, Text } from "react-native";

export default function MatchDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  return (
    <View>
      <Text>Match {id} — detail goes here</Text>
    </View>
  );
}
