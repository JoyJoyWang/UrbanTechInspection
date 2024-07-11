import React, { useState, useEffect } from 'react';

const ImageGallery = () => {
  const [imagePaths, setImagePaths] = useState([]);

  useEffect(() => {
    fetchImagePaths();
  }, []);

  const fetchImagePaths = async () => {
    try {
      const response = await fetch('../../public/images/'); 
      if (!response.ok) {
        throw new Error('Failed to fetch image paths');
      }
      const data = await response.json();
      setImagePaths(data.imagePaths); // Assuming imagePaths is an array of image URLs
    } catch (error) {
      console.error('Error fetching image paths:', error);
    }
  };

  return (
    <div className="image-gallery">
      <h1>Image Gallery</h1>
      <div className="pictures">
        {imagePaths.map((imagePath, index) => (
          <img key={index} src={imagePath} alt={`Image ${index}`} />
        ))}
      </div>
    </div>
  );
};

export default ImageGallery;