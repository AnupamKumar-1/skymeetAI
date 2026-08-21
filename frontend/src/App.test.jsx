import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders landing page brand', () => {
  render(<App />);
  const logo = screen.getByAltText('Hoovik');
  expect(logo).toBeInTheDocument();
});
