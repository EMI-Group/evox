// Run with Node 24+ after building the referenced portal checkout.
// Usage: node docs/tools/export-readme-media.mjs /path/to/evox-portal /path/to/frames
// Requires the portal's installed Playwright, its production build, and Python on PATH.
// Check out the revision recorded in readme-motion-source.json to reproduce an export.
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {stripTypeScriptTypes} from 'node:module';
import {execFileSync,spawn} from 'node:child_process';
const portal=path.resolve(process.argv[2]);
const output=path.resolve(process.argv[3]);
const only=process.argv[4];
const fps=50,frameMs=1000/fps;
const phaseDurations=[1800,4200,480,2200,4200];
const assets=path.resolve(import.meta.dirname,'../source/_static');
const {chromium}=await import(pathToFileURL(path.join(portal,'node_modules/playwright/index.mjs')));
const revision=execFileSync('git',['rev-parse','HEAD'],{cwd:portal,encoding:'utf8'}).trim();
const source=file=>execFileSync('git',['show',`${revision}:${file}`],{cwd:portal,encoding:'utf8'});
const theme=source('src/styles/global.css');
const light=[...theme.matchAll(/:root\[data-theme="light"\] \{([^}]+)\}/g)].map(m=>m[1]).join('\n');
const sceneCode=['src/components/art/agentic-evolution.ts','src/components/art/diagram-theme.ts','src/components/art/evolution-scenes.ts'].map(file=>source(file).replace(/^import .*$/gm,'')).join('\n');
// Rendering-only adaptations preserve the source populations and their scores.
const replaceScene=(text,before,after)=>{if(!text.includes(before))throw new Error('Portal renderer changed; review motion adaptation');return text.replace(before,after);};
let adaptedSceneCode=replaceScene(sceneCode,'at=points[Math.floor(u*50)]','index=u*50,lo=Math.floor(index),fraction=index-lo,at={x:mix(points[lo].x,points[lo+1].x,fraction),y:mix(points[lo].y,points[lo+1].y,fraction)}');
adaptedSceneCode=replaceScene(adaptedSceneCode,'this.codeCard(mix(center.x,right,copy),mix(center.y,rows[i],copy),edits,copy,changed);','this.blendCodeCard(mix(center.x,right,copy),mix(center.y,rows[i],copy),champ.edits,child.edits,edit,copy,changed);');
adaptedSceneCode=replaceScene(adaptedSceneCode,'  agentic(stage:number,p:number) {',`  blendCodeCard(x:number,y:number,from:boolean[],to:boolean[],t:number,alpha:number,changed:number[]) {
    this.frameBox(x,y,112,57,alpha);
    this.text("solve.py",x+8,y+12,this.palette.ink,alpha,9);
    const previous=agentCode(from);
    agentCode(to).forEach((line,i)=>{
      if(!changed.includes(i)){this.text(previous[i],x+8,y+27+i*11,this.palette.muted,alpha,8);return;}
      this.ctx.save();this.ctx.fillStyle=diagramAlpha(this.palette.red,.1*alpha*t);this.ctx.fillRect(x+4,y+18+i*11,104,11);this.ctx.restore();
      this.text(previous[i],x+8,y+27+i*11,this.palette.muted,alpha*(1-t),8);
      this.text(line,x+8,y+27+i*11,this.palette.red,alpha*t,8);
    });
  }
  agentic(stage:number,p:number) {`);
const script=stripTypeScriptTypes(adaptedSceneCode).replace(/^export /gm,'')+'\nwindow.Scene=EvolutionScene;window.nextPopulation=nextPopulation;window.nextAgentPopulation=nextAgentPopulation;window.readPalette=readDiagramPalette;';
const manifest={sourceFiles:['src/components/ParallelDiagram.astro','src/components/art/evolution-scenes.ts','src/components/art/agentic-evolution.ts','src/components/art/diagram-theme.ts','src/styles/global.css','src/i18n/locales/en.json','src/i18n/locales/zh-cn.json'],objectives:'Deterministic illustrative objectives; not measured model, robotics or CFD results.',repository:'https://github.com/EMI-Group/evox-portal',revision,theme:'light',fps,frameDurationMs:frameMs,phaseDurationsMs:phaseDurations,adaptation:'Original populations and scores; interpolated airflow, crossfaded code edits and 180 ms visual phase transitions; localized captions and slightly enlarged labels. Three-step captions group inheritance and variation into Generate. Evaluation holds shortened to 480 ms and code reproduction to 2800 ms; two generations followed by a 400 ms replay crossfade. Hardware captures use the original component with a manual render clock.',scenes:[]};
if(only==='compute'){const previous=JSON.parse(readFileSync(path.join(output,'manifest.json'),'utf8'));if(previous.revision!==revision)throw new Error('Partial export requires the same source revision');manifest.scenes=previous.scenes.filter(scene=>scene.kind!=='compute');}
mkdirSync(output,{recursive:true});
const browser=await chromium.launch({headless:true});let server;
try{
 const page=await browser.newPage({viewport:{width:1100,height:850}});
 await page.setContent(`<style>:root{${light}}</style>`);
 await page.addScriptTag({content:script});
 if(only!=='compute')for(const locale of ['en','zh-cn']){
  const tr=JSON.parse(source(`src/i18n/locales/${locale}.json`));
  for(const kind of ['generative','agentic','physical','science']){
   // Code edits settle sooner than the continuous model, policy and shape changes.
   const durations=kind==='agentic'?[...phaseDurations.slice(0,-1),2800]:phaseDurations;
   const cycleMs=durations.reduce((sum,duration)=>sum+duration,0);
   const sceneMs=cycleMs*2+400,sceneFrames=sceneMs/frameMs;
   const name=`readme-${kind}${locale==='en'?'':'-zh'}`;
   const dir=path.join(output,name);mkdirSync(dir,{recursive:true});
   await page.evaluate(({kind,tr,phaseDurations,cycleMs,sceneMs,frameMs})=>{
    const canvas=document.createElement('canvas');canvas.width=640;canvas.height=374;
    const frame=document.createElement('canvas');frame.width=640;frame.height=432;
    const fc=frame.getContext('2d'),palette=window.readPalette();
    const scene=Object.create(window.Scene.prototype);
    const dataset={generationLabel:tr['scene.generation'],retainedLabel:tr[kind==='generative'?'scene.retained':`scene.${kind}.retained`],testsLabel:tr['scene.agentic.tests'],testEmptyLabel:tr['scene.agentic.test.empty'],testTextLabel:tr['scene.agentic.test.text']};
    const stepKeys=['generate','evaluate','select','generate'];
    const stepNumbers=['01','02','03','01'];
    for(let n=0;n<4;n++)dataset[`caption${n}`]=`${stepNumbers[n]} ${tr[`evolution.step.${stepKeys[n]}`]} · ${tr[`scene.${kind}.${n}`]}`;
    Object.assign(scene,{canvas,ctx:canvas.getContext('2d'),kind,root:{dataset},caption:document.createElement('p'),scale:640/480,ratio:1,palette,durations:phaseDurations,starts:phaseDurations.reduce((times,duration)=>[...times,times.at(-1)+duration],[0])});
    const text=scene.text;scene.text=function(value,x,y,color,alpha=1,size=10){return text.call(this,value,x,y,color,alpha,size+1);};
    const previousVisual=document.createElement('canvas');previousVisual.width=640;previousVisual.height=374;const previousCtx=previousVisual.getContext('2d');
    let lastMs=-Infinity,lastPhase='',blendStart=-Infinity;
    const reset=()=>{lastMs=-Infinity;lastPhase='';blendStart=-Infinity;return Object.assign(scene,{generation:0,time:0,lastCaption:-1,previousPopulation:undefined,offspring:undefined,agentOffspring:undefined,agentPopulation:window.nextAgentPopulation(undefined,0),population:kind==='agentic'?{candidates:[],winner:0}:window.nextPopulation(kind,undefined,0)});};
    reset();
    const draw=ms=>{
     const generation=Math.floor(ms/cycleMs);
     while(scene.generation<generation){
      scene.previousPopulation=scene.population;scene.generation++;
      if(kind==='agentic'){scene.agentPopulation=scene.agentOffspring??window.nextAgentPopulation(scene.agentPopulation,scene.generation);scene.agentOffspring=undefined;}
      else scene.population=scene.offspring??window.nextPopulation(kind,scene.population,scene.generation);
      scene.offspring=undefined;
     }
     scene.time=ms%cycleMs;
     const phase=scene.starts.findIndex((end,index)=>index>0&&scene.time<end)-1,key=`${generation}:${phase}`;
     if(key!==lastPhase&&ms-lastMs<=frameMs+1){
      previousCtx.fillStyle=palette.bg;previousCtx.fillRect(0,0,640,374);previousCtx.drawImage(canvas,0,0);blendStart=ms;
     }
     if(ms-lastMs>frameMs+1)blendStart=-Infinity;
     scene.render();lastMs=ms;lastPhase=key;
     fc.clearRect(0,0,640,432);fc.fillStyle=palette.bg;fc.fillRect(0,0,640,432);fc.drawImage(canvas,0,0);
     if(ms-blendStart<180){const t=Math.max(0,Math.min(1,(ms-blendStart)/180));fc.globalAlpha=1-t*t*(3-2*t);fc.drawImage(previousVisual,0,0);fc.globalAlpha=1;}
     fc.fillStyle=palette.muted;fc.font='18px system-ui, sans-serif';fc.textAlign='center';
     const words=scene.caption.textContent.split(' '),lines=[];let line='';
     for(const word of words){const next=line?line+' '+word:word;if(fc.measureText(next).width>592&&line){lines.push(line);line=word;}else line=next;}lines.push(line);
     lines.forEach((value,i)=>fc.fillText(value,320,392+i*23,600));
     fc.strokeStyle=palette.line;fc.globalAlpha=.3;fc.beginPath();fc.roundRect(.5,.5,639,431,16);fc.stroke();fc.globalAlpha=1;
    };
    draw(0);const first=document.createElement('canvas');first.width=640;first.height=432;first.getContext('2d').drawImage(frame,0,0);
    window.mediaReset=reset;
    window.mediaFrame=ms=>{if(ms<cycleMs*2)draw(ms);else{draw(cycleMs*2-frameMs);fc.globalAlpha=(ms-cycleMs*2+frameMs)/(sceneMs-cycleMs*2);fc.drawImage(first,0,0);fc.globalAlpha=1;}return frame.toDataURL('image/png');};
   },{kind,tr,phaseDurations:durations,cycleMs,sceneMs,frameMs});
   const still=await page.evaluate(ms=>window.mediaFrame(ms),cycleMs-frameMs);writeFileSync(path.join(assets,name+'.png'),Buffer.from(still.split(',')[1],'base64'));
   await page.evaluate(()=>window.mediaReset());
   for(let base=0;base<sceneFrames;base+=20){
    const frames=await page.evaluate(({base,sceneFrames,frameMs})=>Array.from({length:Math.min(20,sceneFrames-base)},(_,j)=>window.mediaFrame((base+j)*frameMs)),{base,sceneFrames,frameMs});
    frames.forEach((data,j)=>writeFileSync(path.join(dir,String(base+j).padStart(4,'0')+'.png'),Buffer.from(data.split(',')[1],'base64')));
   }
   manifest.scenes.push({name,kind,locale,phaseDurationsMs:durations,frames:sceneFrames,durationMs:sceneMs,width:640,height:432});console.log(`${name}: ${sceneFrames} frames`);
  }
 }
 // Capture the actual hardware DOM, SVG and Canvas; only the clock and controls differ.
 server=spawn('python',['-m','http.server','4328','--bind','127.0.0.1','--directory',path.join(portal,'dist')],{stdio:'ignore',windowsHide:true});await new Promise(r=>setTimeout(r,600));
 const component=source('src/components/ParallelDiagram.astro');
 const body=component.match(/<script>([\s\S]*?)<\/script>/)[1].replace(/^\s*import .*$/gm,'').replace('if (!paused) frame = requestAnimationFrame(tick);','').replace('measure(); sync();','measure(); sync(); figure.__readmeSeek = (ms) => { elapsed = ms; draw(); };');
 const hardwareScript=stripTypeScriptTypes(source('src/components/art/diagram-theme.ts')+'\n'+body).replace(/^export /gm,'');
 for(const locale of ['en','zh-cn']){
  await page.goto(`http://127.0.0.1:4328/${locale==='en'?'':locale}`,{waitUntil:'networkidle'});
  await page.evaluate(()=>{document.documentElement.dataset.theme='light';const figure=document.querySelector('.parallel-figure');if(!figure.hasAttribute('data-user-paused'))figure.querySelector('.motion-toggle').click();figure.replaceWith(figure.cloneNode(true));});
  await page.addStyleTag({content:'.parallel-figure{width:960px!important;margin:0!important}.motion-controls{display:none!important}.network-stage{border-radius:16px!important}'});
  await page.addScriptTag({content:hardwareScript});
  const figure=page.locator('.parallel-figure');await figure.scrollIntoViewIfNeeded();await page.evaluate(()=>document.fonts.ready);
  const name=`readme-compute-${locale==='en'?'en':'zh'}`,dir=path.join(output,name);mkdirSync(dir,{recursive:true});
  await figure.evaluate(el=>el.__readmeSeek(10800));await page.emulateMedia({reducedMotion:'reduce'});await page.waitForFunction(()=>!document.querySelector('.parallel-figure').hasAttribute('data-motion-ready'));await figure.screenshot({path:path.join(assets,name+'.png')});await page.emulateMedia({reducedMotion:'no-preference'});await page.waitForFunction(()=>document.querySelector('.parallel-figure').hasAttribute('data-motion-ready'));
  for(let frame=0;frame<12000/frameMs;frame++){await figure.evaluate((el,ms)=>el.__readmeSeek(ms),frame*frameMs);await figure.screenshot({path:path.join(dir,String(frame).padStart(4,'0')+'.png')});}
  const png=readFileSync(path.join(dir,'0000.png'));manifest.scenes.push({name,kind:'compute',locale,frames:12000/frameMs,durationMs:12000,width:png.readUInt32BE(16),height:png.readUInt32BE(20)});console.log(`${name}: ${12000/frameMs} frames`);
 }
 writeFileSync(path.join(output,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');
}finally{await browser.close();server?.kill();}
