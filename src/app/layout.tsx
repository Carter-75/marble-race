import "./globals.css";

import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Marble Race",
  description: "A project named Marble Race",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        {/*
          The following script tags are for Matter.js and Anime.js.
          You can uncomment them and start using them in your project.
          These are powerful animation libraries.
          Feel free to remove them if you don't need them.
        */}
        {/* <script src="https://cdnjs.cloudflare.com/ajax/libs/matter-js/0.19.0/matter.min.js"></script> */}
        {/* <script src="https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.1/anime.min.js"></script> */}
      </head>
      <body>
        {children}
      </body>
    </html>
  );
}

