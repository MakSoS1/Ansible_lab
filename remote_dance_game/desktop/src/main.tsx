import React from 'react'
import ReactDOM from 'react-dom/client'
import { HashRouter, Routes, Route } from 'react-router-dom'
import './styles/index.css'
import Library from './components/Library'
import DanceBuilder from './components/DanceBuilder'
import ConnectPhone from './components/ConnectPhone'
import Gameplay from './components/Gameplay'
import Results from './components/Results'
import Settings from './components/Settings'
import Showcase from './components/Showcase'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <HashRouter>
      <Routes>
        <Route path="/" element={<Library />} />
        <Route path="/build" element={<DanceBuilder />} />
        <Route path="/connect/:danceId" element={<ConnectPhone />} />
        <Route path="/play/:danceId/:sessionId" element={<Gameplay />} />
        <Route path="/results/:sessionId" element={<Results />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/showcase" element={<Showcase />} />
      </Routes>
    </HashRouter>
  </React.StrictMode>,
)
