const object=(properties)=>({type:'object',properties,required:Object.keys(properties),additionalProperties:false});
const choice=(...values)=>({type:'string',enum:values});
const call=(name,args)=>object({tool:choice(name),args:object(args)});
export const actionSchema={oneOf:[
  call('calculate',{op:choice('add','subtract','multiply'),a:{type:'number'},b:{type:'number'}}),
  call('inventory',{}),
  call('gather',{resource:choice('wood','stone'),amount:{type:'integer',minimum:1,maximum:6}}),
  call('craft',{item:choice('shelter','torch')}),
  call('deliver',{parcel:choice('A','B'),route:choice('bridge','tunnel','ferry')}),
  object({final:{type:'string'}}),
]};
