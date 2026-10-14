const BASE=(process.env.NEXT_PUBLIC_API_BASE_URL||"http://localhost:8000").replace(/\/$/,"");
export async function api<T>(path:string,init:RequestInit={},key=""):Promise<T>{
 const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),180000);
 try{const response=await fetch(`${BASE}${path}`,{...init,signal:controller.signal,headers:{...(init.body?{"Content-Type":"application/json"}:{}),...(key?{Authorization:`Bearer ${key}`} :{}),...init.headers}});
 if(!response.ok){const error=await response.json().catch(()=>({}));throw new Error(error.error?.message||`API error ${response.status}`)}return response.json() as Promise<T>;
 }finally{clearTimeout(timer)}
}
export function downloadJSON(value:unknown,name:string){downloadBlob(new Blob([JSON.stringify(value,null,2)],{type:"application/json"}),name)}
export function downloadBlob(blob:Blob,name:string){const url=URL.createObjectURL(blob);const a=document.createElement("a");a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
export async function downloadRun(id:string,key:string){const response=await fetch(`${BASE}/api/v1/experiments/${id}/export`,{headers:key?{Authorization:`Bearer ${key}`}:{}});if(!response.ok)throw new Error("Artifact export failed");downloadBlob(await response.blob(),`experiment-${id}.zip`)}
