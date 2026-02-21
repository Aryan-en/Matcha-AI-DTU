// Status chip — colored badge for COMPLETED / PROCESSING / UPLOADED / FAILED
// TODO: implement
import { View, Text } from "react-native";
import { STATUS_COLORS } from "@/constants/api";

export function StatusChip({ status }: { status: string }) {
  const color = STATUS_COLORS[status as keyof typeof STATUS_COLORS] ?? "#71717a";
  return (
    <View style={{ borderColor: color }}>
      <Text style={{ color }}>{status}</Text>
    </View>
  );
}
