import type {Config,Schema} from "./types";
export function validateConfig(config:Config,schema:Schema):string[]{
 const errors:string[]=[];
 for(const key of Object.keys(config))if(!schema.properties[key])errors.push(`Unknown configuration key: ${key}`);
 for(const [key,rule]of Object.entries(schema.properties)){
 const value=config[key];if(value===undefined){continue}
 if(value===null){if(!rule.anyOf?.some(r=>r.type==="null"))errors.push(`${key} cannot be null`);continue}
 const r=rule.anyOf?.find(p=>p.type!=="null")||rule;
 if(r.enum&&!r.enum.includes(String(value)))errors.push(`${key}: unsupported option`);
 if(r.type==="number"||r.type==="integer"){
 if(typeof value!=="number"||!Number.isFinite(value)||(r.type==="integer"&&!Number.isInteger(value)))errors.push(`${key}: valid ${r.type} required`);
 else if((r.minimum!==undefined&&value<r.minimum)||(r.maximum!==undefined&&value>r.maximum)||(r.exclusiveMinimum!==undefined&&value<=r.exclusiveMinimum)||(r.exclusiveMaximum!==undefined&&value>=r.exclusiveMaximum))errors.push(`${key}: value outside allowed range`);
 }
 if(r.type==="boolean"&&typeof value!=="boolean")errors.push(`${key}: boolean required`);
 if(r.type==="string"&&(typeof value!=="string"||value.length<(r.minLength||0)))errors.push(`${key}: text required`);
 if(r.type==="array"&&!Array.isArray(value))errors.push(`${key}: array required`);
 }
 const layers=config.hidden_layers as number[],dropout=config.dropout as number[];
 if(!Array.isArray(layers)||!layers.length||layers.some(n=>!Number.isInteger(n)||n<2||n>2048))errors.push("hidden_layers: 1 to 6 layers, each with 2 to 2048 neurons");
 if(!Array.isArray(dropout)||dropout.length!==layers?.length||dropout.some(n=>typeof n!=="number"||n<0||n>=1))errors.push("dropout: one value in [0,1) for each layer");
 if((config.features as string[])?.includes("isFraud"))errors.push("Target cannot be a predictor");
 if(Number(config.validation_fraction)+Number(config.test_fraction)>=.7)errors.push("At least 30% must remain in training");
 return errors;
}
export function datasetLabel(meta?:{demonstration:boolean;development_sample:boolean}){return !meta?"Awaiting artifacts":meta.demonstration?"Synthetic demonstration":meta.development_sample?"IEEE-CIS development sample":"IEEE-CIS full labeled data"}
export const groupFields:Record<string,string[]>={
 "Dataset & split":["name","dataset","run_role","expected_fingerprint","max_rows","split","split_seed","validation_fraction","test_fraction","max_memory_mb"],
 "Neural architecture":["hidden_layers","activation","dropout","batch_norm","l1","l2","output"],
 "Training":["optimizer","learning_rate","loss","batch_size","epochs","early_stopping","patience","scheduler","gradient_clip","checkpoint_metric","seed"],
 "Imbalanced learning":["method","fraud_weight","legitimate_weight","smote_strategy","sampling_ratio","neighbors"],
 "Evaluation":["threshold","optimize_threshold","selection_metric","bootstrap_samples"],
 "Tree reference":["tree_iterations","tree_depth","tree_learning_rate","tree_l2","tree_patience","tree_threads","tree_class_weighted","fraud_weight","legitimate_weight","seed","checkpoint_metric","early_stopping"],
 "Feature processing":["features","exclude_features","categorical_features","numerical_imputation","categorical_imputation","encoding","scaling","feature_selection"]
};
