// based on https://andrewwalpole.com/blog/use-vite-for-javascript-libraries/

const path = require('path')
const { defineConfig } = require('vite')

module.exports = defineConfig({
  build: {
    lib: {
      entry: path.resolve(__dirname, 'lib/main.js'),
      name: 'rd_webgpu',
      fileName: (format) => `rd_webgpu.${format}.js`
    },
    minify: false
  }
});
