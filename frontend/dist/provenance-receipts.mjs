export async function rendererReceipt(files,{fetch=globalThis.fetch,crypto=globalThis.crypto}={}){
  const assets={'frontend/dist/app.js':'app.js','frontend/dist/provenance-receipts.mjs':'provenance-receipts.mjs','frontend/dist/index.html':'index.html','frontend/dist/styles.css':'styles.css','frontend/dist/vendor/mermaid-10.9.3.min.js':'vendor/mermaid-10.9.3.min.js'};
  const changed=[],missing=[];
  for(const [key,url] of Object.entries(assets)){
    if(!files?.[key]){missing.push(url);continue;}
    try{
      const response=await fetch(url);if(!response.ok)throw new Error('unavailable');
      const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await response.arrayBuffer()))).map(x=>x.toString(16).padStart(2,'0')).join('');
      if(hash!==files[key])changed.push(url);
    }catch(error){missing.push(url);}
  }
  return {status:changed.length?'DIFFERENT':missing.length?'UNAVAILABLE':'MATCHED',changed,missing};
}

export async function exportReceipt(data, entry, bytes,{fetch=globalThis.fetch,crypto=globalThis.crypto}={}){
  const record=data.viewer_provenance?.execution;
  if(!record||!/^run_[a-f0-9]{32}$/.test(record.run_id||''))return {status:'UNAVAILABLE',reason:'No original export manifest was recorded.'};
  const filename=data.viewer_provenance.manifest_file;
  if(filename!==record.run_id+'.json')return {status:'MISMATCH',reason:'Export manifest identity is inconsistent.'};
  const url='data/provenance/'+filename;
  try{
    const response=await fetch(url);if(!response.ok)throw new Error('Export manifest unavailable.');
    const manifest=await response.json();
    if(manifest.schema_version!=='parliament-run/1'||manifest.run_id!==record.run_id)throw new Error('Export manifest identity mismatch.');
    const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))).map(x=>x.toString(16).padStart(2,'0')).join('');
    if(manifest.artifacts?.[entry.file]!==digest)return {status:'MISMATCH',reason:'Saved viewer JSON differs from its export receipt.',manifest,url};
    return {status:'MATCHED',reason:'This saved JSON matches its finalized export receipt. Source archives have not been verified in the browser.',manifest,url,renderer:await rendererReceipt(data.viewer_provenance?.presentation_files,{fetch,crypto})};
  }catch(error){return {status:'UNAVAILABLE',reason:error.message};}
}

