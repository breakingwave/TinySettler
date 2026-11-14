const SVG_NS='http://www.w3.org/2000/svg';
const palette={WOOD:'#294f3f',BRICK:'#755044',SHEEP:'#596b3f',WHEAT:'#8a7844',ORE:'#515b6b',DESERT:'#75684e'};
const labels={WOOD:'Wood',BRICK:'Brick',SHEEP:'Sheep',WHEAT:'Wheat',ORE:'Ore',DESERT:'Desert'};
const players={RED:'#f07d6b',BLUE:'#79baf0',WHITE:'#f1eee1',ORANGE:'#e6af59'};
function svgEl(name,attributes={},text){const el=document.createElementNS(SVG_NS,name);for(const [key,value] of Object.entries(attributes))if(value!==undefined&&value!==null)el.setAttribute(key,String(value));if(text!==undefined)el.textContent=String(text);return el;}
function normalized(value){return String(value??'').split('.').at(-1).toUpperCase();}
function coords(point){return Array.isArray(point)?point:[point.x,point.y];}
function midpoint(points){return [points.reduce((sum,p)=>sum+p[0],0)/points.length,points.reduce((sum,p)=>sum+p[1],0)/points.length];}
function median(values){const a=values.filter(x=>Number.isFinite(x)&&x>0).sort((a,b)=>a-b);return a[Math.floor(a.length/2)]??1;}

/** Render public board geometry only; no opponent private cards are displayed. */
export function renderBoard(container,state){
  container.replaceChildren();
  const board=state?.board;
  if(!board?.tiles?.length){const placeholder=document.createElement('div');placeholder.className='board-placeholder';const icon=document.createElement('span');icon.textContent='⬡';const title=document.createElement('strong');title.textContent='The island is ready';const note=document.createElement('p');note.textContent='Start a game to see the board, settlements and roads.';placeholder.append(icon,title,note);container.append(placeholder);return;}
  const nodeList=Array.isArray(board.nodes)?board.nodes:Object.entries(board.nodes??{}).map(([id,n])=>({id,...n}));
  const nodeMap=new Map(nodeList.map(n=>[String(n.id),n]));
  const tiles=board.tiles.map(tile=>{
    let points=tile.polygon?.map(coords)??tile.vertices?.map(coords)??tile.nodes?.map(id=>nodeMap.get(String(id))).filter(Boolean).map(coords);
    if(!points?.length&&Number.isFinite(tile.x)&&Number.isFinite(tile.y)){const r=board.hexRadius??1;points=Array.from({length:6},(_,i)=>{const a=Math.PI/180*(30+60*i);return [tile.x+r*Math.cos(a),tile.y+r*Math.sin(a)];});}
    return {...tile,points:points??[]};
  }).filter(tile=>tile.points.length>=3);
  if(!tiles.length){const note=document.createElement('p');note.className='muted';note.textContent='Board geometry is unavailable.';container.append(note);return;}
  const unit=median(tiles.flatMap(t=>t.points.map((p,i)=>Math.hypot(p[0]-t.points[(i+1)%t.points.length][0],p[1]-t.points[(i+1)%t.points.length][1]))));
  const allPoints=tiles.flatMap(t=>t.points),xs=allPoints.map(p=>p[0]),ys=allPoints.map(p=>p[1]);
  const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys),pad=unit*.9;
  const svg=svgEl('svg',{viewBox:`${minX-pad} ${minY-pad} ${maxX-minX+pad*2} ${maxY-minY+pad*2}`,class:'catan-board',role:'img','aria-label':'Catan board with terrain, dice numbers, node identifiers, roads and buildings'});
  svg.append(svgEl('title',{},'Catan board'),svgEl('desc',{},'Terrain labels show resources. The number on each tile is its producing dice roll. Small numbered vertices identify legal building positions. Player-colored lines and houses mark roads and buildings.'));
  const terrain=svgEl('g'),roads=svgEl('g'),vertices=svgEl('g'),harbors=svgEl('g');
  for(const tile of tiles){
    const resource=normalized(tile.resource)||'DESERT',[cx,cy]=midpoint(tile.points);
    const poly=svgEl('polygon',{points:tile.points.map(p=>p.join(',')).join(' '),fill:palette[resource]??palette.DESERT,stroke:'#13231d','stroke-width':unit*.055,'stroke-linejoin':'round'});
    poly.append(svgEl('title',{},`${labels[resource]??resource}${tile.number?' · roll '+tile.number:''} · tile ${tile.id??''}`));terrain.append(poly);
    terrain.append(svgEl('text',{x:cx,y:cy-unit*.35,'text-anchor':'middle',fill:'#f2efdf','font-size':unit*.18,'font-weight':600},labels[resource]??resource));
    const number=Number(tile.number);
    if(Number.isFinite(number)&&number>0){
      terrain.append(svgEl('circle',{cx,cy:cy+unit*.04,r:unit*.255,fill:'#eee7d2',stroke:'#26352a','stroke-width':unit*.025}));
      terrain.append(svgEl('text',{x:cx,y:cy+unit*.095,'text-anchor':'middle',fill:[6,8].includes(number)?'#a53d30':'#253129','font-size':unit*.25,'font-weight':700},number));
      const pips=Math.max(0,6-Math.abs(7-number));for(let i=0;i<pips;i++)terrain.append(svgEl('circle',{cx:cx+(i-(pips-1)/2)*unit*.068,cy:cy+unit*.19,r:unit*.019,fill:[6,8].includes(number)?'#a53d30':'#253129'}));
    }
    const robber=tile.robber||String(board.robber)===String(tile.id);
    if(robber){const g=svgEl('g',{'aria-label':'Robber'});g.append(svgEl('circle',{cx:cx+unit*.42,cy:cy-unit*.07,r:unit*.08,fill:'#111816',stroke:'#e6dfc9','stroke-width':unit*.015}),svgEl('path',{d:`M ${cx+unit*.37} ${cy+unit*.02} L ${cx+unit*.47} ${cy+unit*.02} L ${cx+unit*.52} ${cy+unit*.23} L ${cx+unit*.32} ${cy+unit*.23} Z`,fill:'#111816',stroke:'#e6dfc9','stroke-width':unit*.015}),svgEl('title',{},'Robber blocks this tile'));terrain.append(g);}
  }
  for(const road of board.roads??[]){
    const a=nodeMap.get(String(road.a??road.nodes?.[0])),b=nodeMap.get(String(road.b??road.nodes?.[1]));if(!a||!b)continue;
    const owner=normalized(road.owner??road.color);if(!owner)continue;
    roads.append(svgEl('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:'#121b15','stroke-width':unit*.15,'stroke-linecap':'round'}));
    const line=svgEl('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:players[owner]??'#c2e68b','stroke-width':unit*.09,'stroke-linecap':'round'});line.append(svgEl('title',{},`${owner} road ${a.id}–${b.id}`));roads.append(line);
  }
  for(const node of nodeList){
    if(!Number.isFinite(node.x)||!Number.isFinite(node.y))continue;
    const owner=normalized(node.owner??node.color),building=normalized(node.building??node.type);const x=node.x,y=node.y;
    vertices.append(svgEl('circle',{cx:x,cy:y,r:unit*.075,fill:'#17221c',stroke:'#a7b298','stroke-width':unit*.012}));
    if(owner){
      const city=building.includes('CITY'),r=unit*(city ? .18 : .13);
      const shape=city?`M ${x-r} ${y+r*.55} V ${y-r*.9} H ${x} V ${y-r*.3} H ${x+r} V ${y+r*.55} Z`:`M ${x-r} ${y+r*.6} V ${y-r*.1} L ${x} ${y-r} L ${x+r} ${y-r*.1} V ${y+r*.6} Z`;
      const piece=svgEl('path',{d:shape,fill:players[owner]??'#c2e68b',stroke:'#111a14','stroke-width':unit*.035,'stroke-linejoin':'round'});piece.append(svgEl('title',{},`${owner} ${city?'city':'settlement'} · node ${node.id}`));vertices.append(piece);
    }
    const label=svgEl('text',{x:x+unit*.105,y:y+unit*.17,'font-size':unit*.115,fill:'#dce3cf','font-weight':500,stroke:'#18231c','stroke-width':unit*.035,'paint-order':'stroke','stroke-linejoin':'round'},node.id);vertices.append(label);
  }
  const boardCenter=[(minX+maxX)/2,(minY+maxY)/2];
  // Earlier saved game snapshots grouped equal-resource harbors together.
  // Recover their adjacent pairs without changing the saved game or game rules.
  const ports=(board.ports??[]).flatMap(port=>{
    if((port.nodes?.length??0)<=2)return [port];
    const members=new Set(port.nodes.map(String)),used=new Set(),pairs=[];
    for(const edge of board.edges??[]){const a=String(edge.a),b=String(edge.b);if(members.has(a)&&members.has(b)&&!used.has(a)&&!used.has(b)){pairs.push({...port,nodes:[edge.a,edge.b]});used.add(a);used.add(b);}}
    return pairs;
  });
  for(const port of ports){
    const points=(port.nodes??[]).map(id=>nodeMap.get(String(id))).filter(Boolean).map(coords);if(!points.length)continue;
    const [mx,my]=midpoint(points),length=Math.hypot(mx-boardCenter[0],my-boardCenter[1])||1;
    const px=mx+(mx-boardCenter[0])/length*unit*.44,py=my+(my-boardCenter[1])/length*unit*.44;
    const resource=normalized(port.resource),label=labels[resource]?`${labels[resource]} 2:1`:'Any 3:1';
    harbors.append(svgEl('line',{x1:mx,y1:my,x2:px,y2:py,stroke:'#728b7d','stroke-width':unit*.025}),svgEl('rect',{x:px-unit*.34,y:py-unit*.105,width:unit*.68,height:unit*.21,rx:unit*.045,fill:'#14241f',stroke:'#6b8575','stroke-width':unit*.012}),svgEl('text',{x:px,y:py+unit*.036,'font-size':unit*.105,'text-anchor':'middle',fill:'#c3d1be'},label));
  }
  svg.append(terrain,roads,harbors,vertices);container.append(svg);
  const legend=document.createElement('div');legend.className='board-legend';for(const [name,color] of Object.entries(players)){const chip=document.createElement('span'),dot=document.createElement('i');dot.style.background=color;chip.append(dot,document.createTextNode(name==='RED'?'Model · Red':name[0]+name.slice(1).toLowerCase()));legend.append(chip);}const note=document.createElement('span');note.className='board-key';note.textContent='Vertex numbers = node IDs · dots = roll likelihood';legend.append(note);container.append(legend);
}
