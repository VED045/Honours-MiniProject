import {describe,it,expect,vi,afterEach} from "vitest";
import {render,screen,cleanup} from "@testing-library/react";
import React from "react";
import {Confusion,ModeBadge} from "../components/ui";
import {datasetLabel,validateConfig} from "../lib/config";
import {api} from "../lib/api";
import type {Metrics} from "../lib/types";
afterEach(()=>{cleanup();vi.unstubAllGlobals()});
describe("honest model UI",()=>{
 it("labels synthetic and development artifacts",()=>{expect(datasetLabel({demonstration:true,development_sample:false})).toBe("Synthetic demonstration");render(<ModeBadge label={datasetLabel({demonstration:true,development_sample:false})}/>);expect(screen.getByText("Synthetic demonstration")).toBeTruthy();expect(datasetLabel({demonstration:false,development_sample:true})).toContain("development sample")});
 it("updates confusion cells when threshold-derived metrics change",()=>{const first={tn:30,fp:10,fn:4,tp:6} as Metrics;const {rerender}=render(<Confusion metrics={first}/>);expect(screen.getByText("10")).toBeTruthy();rerender(<Confusion metrics={{...first,fp:5,fn:7,tp:3,tn:35}}/>);expect(screen.getByText("5")).toBeTruthy();expect(screen.queryByText("10")).toBeNull()});
 it("rejects invalid hyperparameters and target leakage",()=>{const errors=validateConfig({hidden_layers:[64,32],dropout:[1],features:["isFraud"],validation_fraction:.4,test_fraction:.4,epochs:0},{properties:{epochs:{type:"integer",minimum:1,maximum:500}}});expect(errors.some(e=>e.includes("epochs"))).toBe(true);expect(errors.some(e=>e.includes("dropout"))).toBe(true);expect(errors.some(e=>e.includes("Target"))).toBe(true)});
 it("exposes backend failures instead of returning synthetic output",async()=>{vi.stubGlobal("fetch",vi.fn().mockResolvedValue({ok:false,status:409,json:async()=>({error:{message:"Dataset unavailable"}})}));await expect(api("/api/v1/dataset/summary")).rejects.toThrow("Dataset unavailable")});
});
