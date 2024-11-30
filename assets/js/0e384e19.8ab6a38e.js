"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[976],{2053:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>c,contentTitle:()=>a,default:()=>h,frontMatter:()=>l,metadata:()=>i,toc:()=>d});const i=JSON.parse('{"id":"intro","title":"Quickstart","description":"trolo is a framework for harnessing the power of transformers with YOLO models and other single-shot detectors!","source":"@site/docs/intro.md","sourceDirName":".","slug":"/intro","permalink":"/trolo/intro","draft":false,"unlisted":false,"editUrl":"https://github.com/ayushexel/trolo/tree/main/docs/docs/intro.md","tags":[],"version":"current","sidebarPosition":1,"frontMatter":{"sidebar_position":1},"sidebar":"tutorialSidebar","next":{"title":"index","permalink":"/trolo/models/"}}');var r=t(4848),s=t(8453);const l={sidebar_position:1},a="Quickstart",c={},d=[{value:"Installation",id:"installation",level:2},{value:"Key Features",id:"key-features",level:2},{value:"Available Models",id:"available-models",level:2},{value:"\ud83d\udd25 NEW \ud83d\udd25 D-FINE",id:"-new--d-fine",level:3},{value:"CLI Interface",id:"cli-interface",level:3},{value:"Python API",id:"python-api",level:3},{value:"CLI Interface",id:"cli-interface-1",level:3},{value:"Python API",id:"python-api-1",level:3}];function o(e){const n={a:"a",code:"code",h1:"h1",h2:"h2",h3:"h3",header:"header",img:"img",li:"li",p:"p",pre:"pre",table:"table",tbody:"tbody",td:"td",th:"th",thead:"thead",tr:"tr",ul:"ul",...(0,s.R)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(n.header,{children:(0,r.jsx)(n.h1,{id:"quickstart",children:"Quickstart"})}),"\n",(0,r.jsx)(n.p,{children:"trolo is a framework for harnessing the power of transformers with YOLO models and other single-shot detectors!"}),"\n",(0,r.jsx)(n.h2,{id:"installation",children:"Installation"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-bash",children:"pip install trolo\n"})}),"\n",(0,r.jsx)(n.h2,{id:"key-features",children:"Key Features"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsx)(n.li,{children:"\ud83d\udd25 Transformer-enhanced object detection"}),"\n",(0,r.jsx)(n.li,{children:"\ud83c\udfaf Single-shot detection capabilities"}),"\n",(0,r.jsx)(n.li,{children:"\u26a1 High performance inference"}),"\n",(0,r.jsx)(n.li,{children:"\ud83d\udee0\ufe0f Easy to use CLI interface"}),"\n",(0,r.jsx)(n.li,{children:"\ud83d\ude80 Fast video stream inference"}),"\n",(0,r.jsx)(n.li,{children:"\ud83e\udde0 Automatic DDP handling"}),"\n"]}),"\n",(0,r.jsx)(n.h2,{id:"available-models",children:"Available Models"}),"\n",(0,r.jsx)(n.p,{children:"trolo provides several state-of-the-art detection models:"}),"\n",(0,r.jsx)(n.h3,{id:"-new--d-fine",children:"\ud83d\udd25 NEW \ud83d\udd25 D-FINE"}),"\n",(0,r.jsxs)(n.p,{children:["The D-FINE model redefines regression tasks in DETR-based detectors using Fine-grained Distribution Refinement (FDR).\n",(0,r.jsx)(n.a,{href:"https://arxiv.org/abs/2410.13842",children:"Official Paper"})," | ",(0,r.jsx)(n.a,{href:"https://github.com/Peterande/D-FINE",children:"Official Implementation"})]}),"\n",(0,r.jsx)(n.p,{children:(0,r.jsx)(n.img,{src:"https://raw.githubusercontent.com/Peterande/storage/master/figs/stats_padded.png",alt:"D-FINE Stats"})}),"\n",(0,r.jsxs)(n.table,{children:[(0,r.jsx)(n.thead,{children:(0,r.jsxs)(n.tr,{children:[(0,r.jsx)(n.th,{style:{textAlign:"center"},children:"Model"}),(0,r.jsxs)(n.th,{style:{textAlign:"center"},children:["AP",(0,r.jsx)("sup",{children:"val"})]}),(0,r.jsx)(n.th,{style:{textAlign:"center"},children:"Size"}),(0,r.jsx)(n.th,{style:{textAlign:"center"},children:"Latency"})]})}),(0,r.jsxs)(n.tbody,{children:[(0,r.jsxs)(n.tr,{children:[(0,r.jsx)(n.td,{style:{textAlign:"center"},children:"dfine-n"}),(0,r.jsx)(n.td,{style:{textAlign:"center"},children:"42.8"}),(0,r.jsx)(n.td,{style:{textAlign:"center"},children:"4M"}),(0,r.jsx)(n.td,{style:{textAlign:"center"},children:"2.12ms"})]}),(0,r.jsxs)(n.tr,{children:[(0,r.jsx)(n.td,{style:{textAlign:"center"},children:"dfine-s"}),(0,r.jsx)(n.td,{style:{textAlign:"center"},children:"48.5"}),(0,r.jsx)(n.td,{style:{textAlign:"center"},children:"10M"}),(0,r.jsx)(n.td,{style:{textAlign:"center"},children:"3.49ms"})]}),(0,r.jsxs)(n.tr,{children:[(0,r.jsx)(n.td,{style:{textAlign:"center"},children:"dfine-m"}),(0,r.jsx)(n.td,{style:{textAlign:"center"},children:"52.3"}),(0,r.jsx)(n.td,{style:{textAlign:"center"},children:"19M"}),(0,r.jsx)(n.td,{style:{textAlign:"center"},children:"5.62ms"})]})]})]}),"\n",(0,r.jsxs)(n.p,{children:["Find all the available models ",(0,r.jsx)(n.a,{href:"/trolo/models/",children:"here"}),"."]}),"\n",(0,r.jsx)(n.h1,{id:"inference",children:"Inference"}),"\n",(0,r.jsx)(n.h3,{id:"cli-interface",children:"CLI Interface"}),"\n",(0,r.jsx)(n.p,{children:"The basic command structure is:"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-bash",children:"trolo [command] [options]\n"})}),"\n",(0,r.jsx)(n.p,{children:"For help:"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-bash",children:"trolo --help # general help\ntrolo [command] --help # command-specific help\n"})}),"\n",(0,r.jsx)(n.h3,{id:"python-api",children:"Python API"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:'from trolo.inference import DetectionPredictor\n\npredictor = DetectionPredictor(model="dfine-n")\npredictions = predictor.predict() # get predictions\nplotted_preds = predictor.visualize(show=True, save=True) # visualize outputs\n'})}),"\n",(0,r.jsxs)(n.p,{children:["Visit the ",(0,r.jsx)(n.a,{href:"/features/inference",children:"inference"})," section for detailed usage instructions."]}),"\n",(0,r.jsx)(n.h1,{id:"training",children:"Training"}),"\n",(0,r.jsx)(n.h3,{id:"cli-interface-1",children:"CLI Interface"}),"\n",(0,r.jsx)(n.p,{children:"The basic command structure is:"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-bash",children:"trolo train [options]\n"})}),"\n",(0,r.jsx)(n.p,{children:"Basic training examples:"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-bash",children:"trolo train --config dfine_n  # train using built-in config\ntrolo train --model dfine-n --dataset coco  # specify model and dataset separately\n"})}),"\n",(0,r.jsx)(n.p,{children:"\ud83d\udd25 Automatic multi-GPU handling. Just specify the devices:"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-bash",children:"trolo train --device 0,1,2,3  # multi-GPU training\n"})}),"\n",(0,r.jsx)(n.h3,{id:"python-api-1",children:"Python API"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:'from trolo.trainers import DetectionTrainer\n\n# Initialize trainer\ntrainer = DetectionTrainer(\n    config="dfine_n"\n)\n\n# Start training\ntrainer.fit()  # single GPU\ntrainer.fit(device="0,1,2,3")  # multi-GPU\n'})}),"\n",(0,r.jsxs)(n.p,{children:["Visit the ",(0,r.jsx)(n.a,{href:"/features/training",children:"Training"})," section for detailed configuration options."]})]})}function h(e={}){const{wrapper:n}={...(0,s.R)(),...e.components};return n?(0,r.jsx)(n,{...e,children:(0,r.jsx)(o,{...e})}):o(e)}},8453:(e,n,t)=>{t.d(n,{R:()=>l,x:()=>a});var i=t(6540);const r={},s=i.createContext(r);function l(e){const n=i.useContext(s);return i.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function a(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(r):e.components||r:l(e.components),i.createElement(s.Provider,{value:n},e.children)}}}]);