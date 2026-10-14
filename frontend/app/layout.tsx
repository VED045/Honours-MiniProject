import type { Metadata } from "next";
import "./globals.css";
import {BackendProvider} from "../components/BackendContext";
export const metadata: Metadata={ title:"Vesta Lab | Financial Fraud Research",description:"Reproducible ANN experiments with imbalanced learning on IEEE-CIS" };
export default function RootLayout({children}:{children:React.ReactNode}){return <html lang="en"><body><BackendProvider>{children}</BackendProvider></body></html>}
