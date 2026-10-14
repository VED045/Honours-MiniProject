"use client";
import {createContext,useContext,useState} from "react";
const BackendContext=createContext<{key:string;setKey:(key:string)=>void}>({key:"",setKey:()=>{}});
export function BackendProvider({children}:{children:React.ReactNode}){const [key,setKey]=useState("");return <BackendContext.Provider value={{key,setKey}}>{children}</BackendContext.Provider>}
export function useBackend(){return useContext(BackendContext)}
