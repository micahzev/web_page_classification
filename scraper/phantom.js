// Simple Javascript example

console.log('Loading a web page');
var page = require('webpage').create();
var url = 'http://phantomjs.org/';
page.open(url, function (status) {
  // page.evaluate(function() {
  //   console.log(document.title);
  // });
  phantom.exit();
});
      -