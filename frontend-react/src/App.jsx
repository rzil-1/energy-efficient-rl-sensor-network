import { useEffect, useRef, useState } from 'react'
import './App.css'

function App() {
  const canvasRef = useRef(null)
  const [terrain, setTerrain] = useState(null)
  const [simulationState, setSimulationState] = useState({ step: 0, sensors: [] })
  const [isRunning, setIsRunning] = useState(false)
  const [eventLogs, setEventLogs] = useState([])
  const [showModal, setShowModal] = useState(false)

  // Derived stats
  const activeSensors = simulationState.sensors.filter(s => s.battery > 0)
  const avgBattery = activeSensors.length > 0 
    ? activeSensors.reduce((sum, s) => sum + s.battery, 0) / activeSensors.length 
    : 0;
  
  // Calculate approximate coverage area (normalized radius squared)
  const maxPossibleCoverage = simulationState.sensors.length * (0.2 * 0.2 * Math.PI); // Assuming 0.2 max radius
  const currentCoverage = activeSensors.reduce((sum, s) => sum + (s.radius * s.radius * Math.PI), 0);
  const coveragePercent = maxPossibleCoverage > 0 ? Math.min(100, (currentCoverage / maxPossibleCoverage) * 100) : 0;

  // Load terrain
  useEffect(() => {
    fetch('http://localhost:8000/api/terrain')
      .then(r => r.json())
      .then(data => {
        const img = new Image()
        img.onload = () => setTerrain(img)
        img.src = data.image
      })
  }, [])

  // Draw loop
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    canvas.width = window.innerWidth
    canvas.height = window.innerHeight

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    // Draw terrain
    if (terrain) {
        // Darken terrain to make neon pop
        ctx.globalCompositeOperation = 'source-over';
        ctx.drawImage(terrain, 0, 0, canvas.width, canvas.height)
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)'
        ctx.fillRect(0, 0, canvas.width, canvas.height)
    }

    // Enable additive blending for glowing effect
    ctx.globalCompositeOperation = 'lighter';

    // Draw sensors
    simulationState.sensors.forEach(s => {
        const px = s.x * canvas.width
        const py = s.y * canvas.height
        const rad = s.radius * Math.min(canvas.width, canvas.height)

        if (s.battery > 0) {
            // Determine neon color
            let r, g, b;
            if (s.battery > 0.7) { r=0; g=255; b=128; } // Neon Green
            else if (s.battery > 0.3) { r=255; g=234; b=0; } // Neon Yellow
            else { r=255; g=51; b=102; } // Neon Red

            const colorStr = `${r}, ${g}, ${b}`;

            // Draw coverage glow (large circle)
            ctx.beginPath()
            ctx.arc(px, py, rad, 0, Math.PI * 2)
            ctx.fillStyle = `rgba(${colorStr}, 0.15)`
            ctx.fill()
            ctx.lineWidth = 1;
            ctx.strokeStyle = `rgba(${colorStr}, 0.5)`
            ctx.stroke()
            
            // Draw core node
            ctx.beginPath()
            ctx.arc(px, py, 4, 0, Math.PI * 2)
            ctx.fillStyle = `rgb(${colorStr})`
            ctx.shadowColor = `rgb(${colorStr})`
            ctx.shadowBlur = 15
            ctx.fill()
            ctx.shadowBlur = 0 // reset
        } else {
            // Dead node
            ctx.globalCompositeOperation = 'source-over';
            ctx.beginPath()
            ctx.arc(px, py, 3, 0, Math.PI * 2)
            ctx.fillStyle = '#333'
            ctx.fill()
            ctx.globalCompositeOperation = 'lighter';
        }
    })
    
    // Reset blend mode
    ctx.globalCompositeOperation = 'source-over';

  }, [terrain, simulationState])

  const prevDeadCountRef = useRef(0)
  const prevBatteryDropRef = useRef(1.0)

  const startSimulation = () => {
    setIsRunning(true)
    setEventLogs([{step: 0, msg: "Simulation Started"}])
    prevDeadCountRef.current = 0
    prevBatteryDropRef.current = 1.0

    const ws = new WebSocket('ws://localhost:8000/ws/simulate')
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        setSimulationState(data)
        
        // Compute stats for logging
        const deadCount = data.sensors.filter(s => s.battery <= 0.01).length;
        const activeSensors = data.sensors.filter(s => s.battery > 0.01);
        const avgBattery = activeSensors.length > 0 
          ? activeSensors.reduce((sum, s) => sum + s.battery, 0) / activeSensors.length 
          : 0;

        const newLogs = [];

        // Log when sensors die
        if (deadCount > prevDeadCountRef.current) {
            const diff = deadCount - prevDeadCountRef.current;
            newLogs.push({ step: data.step, msg: `ALERT: ${diff} sensor(s) went offline.` });
            prevDeadCountRef.current = deadCount;
        }

        // Log battery milestone drops (every 10%)
        if (avgBattery < prevBatteryDropRef.current - 0.1) {
            newLogs.push({ step: data.step, msg: `Network average battery dropped below ${(prevBatteryDropRef.current * 100 - 10).toFixed(0)}%.` });
            prevBatteryDropRef.current -= 0.1;
        }

        // Periodic ping
        if (data.step > 0 && data.step % 100 === 0) {
            newLogs.push({ step: data.step, msg: `Routine Check: ${activeSensors.length} active, avg battery ${(avgBattery * 100).toFixed(1)}%.` });
        }

        if (newLogs.length > 0) {
            setEventLogs(prev => [...newLogs.reverse(), ...prev].slice(0, 50));
        }
    }
    ws.onclose = () => setIsRunning(false)
  }

  return (
    <div className="app-container">
      <canvas ref={canvasRef} />
      
      <div className="hud-panel left-panel">
        <h2>Network Command</h2>
        <h3>Global Status</h3>
        
        <div className="stat-row">
          <span className="stat-label">Step</span>
          <span className="stat-value">{simulationState.step} / 500</span>
        </div>
        <div className="stat-row">
          <span className="stat-label">Active Nodes</span>
          <span className="stat-value" style={{color: activeSensors.length < 25 ? '#ff3366' : '#00ff80'}}>
            {activeSensors.length} / {simulationState.sensors.length}
          </span>
        </div>
        <div className="stat-row">
          <span className="stat-label">Avg Battery</span>
          <span className="stat-value" style={{color: avgBattery > 0.7 ? '#00ff80' : avgBattery > 0.3 ? '#ffea00' : '#ff3366'}}>
            {(avgBattery * 100).toFixed(1)}%
          </span>
        </div>
        <div className="stat-row">
          <span className="stat-label">Est. Coverage</span>
          <span className="stat-value">
            {coveragePercent.toFixed(1)}%
          </span>
        </div>

        <button 
          style={{marginTop: '2rem', width: '100%', fontSize: '0.9rem'}} 
          onClick={() => setShowModal(true)}>
          View Model Comparison
        </button>
      </div>

      <div className="hud-panel right-panel">
        <h3>Live Event Log</h3>
        <div className="event-log">
          {eventLogs.map((log, i) => (
            <div key={i} className="event-item">
              <span style={{color: '#888'}}>[{log.step.toString().padStart(3, '0')}]</span> {log.msg}
            </div>
          ))}
        </div>
      </div>

      <div className="bottom-bar">
        <button onClick={startSimulation} disabled={isRunning}>
            {isRunning ? 'Simulating...' : 'INITIATE SIMULATION'}
        </button>
      </div>

      {showModal && (
        <div className="modal-overlay" onClick={() => setShowModal(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <img src="/comparison.png" alt="Model Comparison" />
            <button className="close-btn" onClick={() => setShowModal(false)}>Close</button>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
