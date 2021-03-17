import React from 'react';
import { BrowserRouter as Router, Switch, Route } from 'react-router-dom';
import './App.global.css';

const MainView = () => {
  return (
    <div>
      <p>Hello world</p>
      <button type="button" className="btn btn-primary">
        Click me
      </button>
    </div>
  );
};

export default function App() {
  return (
    <Router>
      <Switch>
        <Route path="/" component={MainView} />
      </Switch>
    </Router>
  );
}
