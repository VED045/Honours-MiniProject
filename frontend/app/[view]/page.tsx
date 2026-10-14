import Dashboard from "../../components/Dashboard";
import { notFound } from "next/navigation";
export async function generateStaticParams(){return ["laboratory","comparison","analyzer","explainability","dataset","history","research"].map(view=>({view}))}
export default async function Page({params}:{params:Promise<{view:string}>}){const {view}=await params;if(!["laboratory","comparison","analyzer","explainability","dataset","history","research"].includes(view))notFound();return <Dashboard initialView={view}/>}
