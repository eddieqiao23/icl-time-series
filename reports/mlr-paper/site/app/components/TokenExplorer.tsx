"use client";

import { useState } from "react";

const taskCount = 6;
const components = ["A", "B", "A", "B", "A", "A"];

export default function TokenExplorer() {
  const [supportCount, setSupportCount] = useState(3);
  const [selectedTask, setSelectedTask] = useState(taskCount - 1);
  const inputPosition = 2 * selectedTask;

  return (
    <div className="token-explorer">
      <div className="token-controls">
        <label htmlFor="support-count">Support pairs</label>
        <select id="support-count" value={supportCount}
          onChange={(event) => setSupportCount(Number(event.target.value))}>
          {[2, 3, 4, 5].map((count) => <option key={count} value={count}>T = {count}</option>)}
        </select>
        <span>Select an input token to inspect its causal context.</span>
      </div>

      <div className="token-sequence" aria-label="Alternating task input and query output tokens">
        {Array.from({ length: taskCount }, (_, task) => {
          const inputPos = 2 * task;
          const outputPos = inputPos + 1;
          const past = task < selectedTask;
          const current = task === selectedTask;
          const future = task > selectedTask;
          return (
            <div className={`token-pair ${future ? "future" : ""}`} key={task}>
              <button type="button" className={`packed-token component-${components[task].toLowerCase()} ${current ? "selected" : ""}`}
                aria-pressed={current} onClick={() => setSelectedTask(task)}>
                <span>x token · task {task}</span><small>position {inputPos} · component {components[task]}</small>
              </button>
              <div className={`label-token ${current ? "masked" : ""} ${past ? "visible" : ""}`}>
                <span>y<sub>{task}</sub></span><small>position {outputPos}</small>
              </div>
            </div>
          );
        })}
      </div>

      <div className="token-detail" aria-live="polite">
        <div><span>Selected input</span><strong>task {selectedTask} · position {inputPosition}</strong></div>
        <div><span>Packed fields</span><strong>{Array.from({ length: supportCount }, (_, i) => `(x${i + 1}, y${i + 1})`).join("  ")} · x<sub>query</sub></strong></div>
        <div><span>Can attend to</span><strong>positions 0–{inputPosition} · not y<sub>{selectedTask}</sub> or future tokens</strong></div>
      </div>
      <p className="token-readout">The hidden state at input position {inputPosition} predicts y<sub>{selectedTask}</sub>. For the paper’s final task, this is position 98 attending over 49 earlier task pairs plus itself.</p>
    </div>
  );
}
