import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SWB 对局模拟器",
  description: "在本地与训练完成的 SWB PPO 策略进行完整对局。",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
