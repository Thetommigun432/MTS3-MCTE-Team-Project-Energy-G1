/**
 * MSW server setup for Node.js test environment.
 * 
 * This creates the mock server that intercepts HTTP requests during tests.
 */
import { setupServer } from 'msw/node';
import { handlers } from './handlers';

// Create the server with default handlers
export const server = setupServer(...handlers);
