const {createProxyMiddleware} = require('http-proxy-middleware');

module.exports = function(app) {
  app.use(createProxyMiddleware('/api',
      {
        "target": 'http://127.0.0.1:5000/',
        "secure": false,
        "logLevel": "debug",
        changeOrigin: true
      }));
  app.use(createProxyMiddleware('/appdata',
      {
        "target": 'http://127.0.0.1:5000/',
        "secure": false,
        "logLevel": "debug",
        changeOrigin: true
      }));
};