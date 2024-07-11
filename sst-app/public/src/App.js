// import React, { useState } from 'react';
// import './App.css';
// import Header from './components/Header';
// import About from './components/About';
// import Product from './components/Product';
// import Contact from './components/Contact';
// import Viewer from './components/viewer.js';

// // View a single image
// const singleImageViewer = new Viewer(document.getElementById('image'), {
//     inline: false, // Set to true if you want it inline
//     viewed() {
//       singleImageViewer.zoomTo(1);
//     },
//   });
  
//   // View a list of images
//   const galleryViewer = new Viewer(document.getElementById('images'), {
//     inline: false, // Set to true if you want it inline
//     viewed() {
//       galleryViewer.zoomTo(1);
//     },
//   });

// function App() {
//     const [activeTab, setActiveTab] = useState('About');

//     return (
//         <div className="App">
//             <Header setActiveTab={setActiveTab} />
//             {activeTab === 'About' && <About />}
//             {activeTab === 'Product' && <Product setActiveTab={setActiveTab} />}
//             {activeTab === 'Contact' && <Contact />}
//         </div>
//     );
// }

// export default App;
