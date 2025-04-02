import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import Select from 'react-select';
import apiService from '../services/api';

const VectorVisualization = () => {
  // State for visualization data and parameters
  const [visualizationData, setVisualizationData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedPoint, setSelectedPoint] = useState(null);
  
  // Visualization parameters
  const [method, setMethod] = useState({ value: 'tsne', label: 't-SNE' });
  const [perplexity, setPerplexity] = useState(30);
  const [nNeighbors, setNNeighbors] = useState(15);
  const [minDist, setMinDist] = useState(0.1);
  const [filterText, setFilterText] = useState('');
  
  // Refs for D3 visualization
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);
  
  // Method options for dropdown
  const methodOptions = [
    { value: 'tsne', label: 't-SNE' },
    { value: 'umap', label: 'UMAP' },
    { value: 'pca', label: 'PCA' }
  ];
  
  // Fetch visualization data
  const fetchVisualizationData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Prepare parameters based on selected method
      const params = {
        method: method.value,
        random_state: 42
      };
      
      // Add method-specific parameters
      if (method.value === 'tsne') {
        params.perplexity = perplexity;
      } else if (method.value === 'umap') {
        params.n_neighbors = nNeighbors;
        params.min_dist = minDist;
      }
      
      const data = await apiService.getVisualizationData(params);
      setVisualizationData(data);
    } catch (err) {
      console.error('Error fetching visualization data:', err);
      setError('Failed to fetch visualization data. Please ensure you have documents uploaded.');
    } finally {
      setLoading(false);
    }
  };
  
  // Initialize visualization on component mount
  useEffect(() => {
    fetchVisualizationData();
  }, []);
  
  // Update visualization when parameters change
  useEffect(() => {
    if (visualizationData) {
      renderVisualization();
    }
  }, [visualizationData, filterText, selectedPoint]);
  
  // Render D3 visualization
  const renderVisualization = () => {
    if (!svgRef.current || !visualizationData) return;
    
    // Clear previous visualization
    d3.select(svgRef.current).selectAll('*').remove();
    
    // Filter points based on filterText
    const filteredPoints = visualizationData.points.filter(point => {
      if (!filterText) return true;
      
      const metadata = point.metadata || {};
      const title = metadata.title || '';
      const filename = metadata.filename || '';
      const content = metadata.content_preview || '';
      
      return (
        title.toLowerCase().includes(filterText.toLowerCase()) ||
        filename.toLowerCase().includes(filterText.toLowerCase()) ||
        content.toLowerCase().includes(filterText.toLowerCase())
      );
    });
    
    if (filteredPoints.length === 0) {
      // Display message if no points match filter
      d3.select(svgRef.current)
        .append('text')
        .attr('x', 400)
        .attr('y', 300)
        .attr('text-anchor', 'middle')
        .text('No documents match the filter criteria');
      return;
    }
    
    // Set up SVG dimensions
    const width = 800;
    const height = 600;
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Create SVG element
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);
    
    // Create a group for the visualization
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Extract x and y coordinates
    const xValues = filteredPoints.map(d => d.x);
    const yValues = filteredPoints.map(d => d.y);
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([d3.min(xValues) * 1.1, d3.max(xValues) * 1.1])
      .range([0, innerWidth]);
    
    const yScale = d3.scaleLinear()
      .domain([d3.min(yValues) * 1.1, d3.max(yValues) * 1.1])
      .range([innerHeight, 0]);
    
    // Create a color scale based on metadata if available
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
    
    // Create a zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.5, 10])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });
    
    // Apply zoom behavior to SVG
    svg.call(zoom);
    
    // Create axes
    const xAxis = d3.axisBottom(xScale);
    const yAxis = d3.axisLeft(yScale);
    
    // Add x-axis
    g.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis);
    
    // Add y-axis
    g.append('g')
      .attr('class', 'y-axis')
      .call(yAxis);
    
    // Add axis labels
    g.append('text')
      .attr('class', 'x-label')
      .attr('x', innerWidth / 2)
      .attr('y', innerHeight + margin.bottom - 5)
      .attr('text-anchor', 'middle')
      .text('Dimension 1');
    
    g.append('text')
      .attr('class', 'y-label')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerHeight / 2)
      .attr('y', -margin.left + 10)
      .attr('text-anchor', 'middle')
      .text('Dimension 2');
    
    // Create tooltip
    const tooltip = d3.select(tooltipRef.current);
    
    // Add points
    g.selectAll('.point')
      .data(filteredPoints)
      .enter()
      .append('circle')
      .attr('class', 'point')
      .attr('cx', d => xScale(d.x))
      .attr('cy', d => yScale(d.y))
      .attr('r', d => selectedPoint && d.id === selectedPoint.id ? 8 : 5)
      .attr('fill', d => {
        // Use file type for color if available
        if (d.metadata && d.metadata.file_type) {
          return colorScale(d.metadata.file_type);
        }
        return colorScale(0);
      })
      .attr('stroke', d => selectedPoint && d.id === selectedPoint.id ? '#000' : 'none')
      .attr('stroke-width', 2)
      .attr('opacity', 0.7)
      .on('mouseover', (event, d) => {
        // Show tooltip on hover
        const title = d.metadata && d.metadata.title ? d.metadata.title : 'Document';
        
        tooltip
          .style('opacity', 1)
          .style('left', `${event.pageX + 10}px`)
          .style('top', `${event.pageY - 10}px`)
          .html(`
            <strong>${title}</strong><br/>
            ${d.metadata && d.metadata.file_type ? `Type: ${d.metadata.file_type}` : ''}
          `);
      })
      .on('mouseout', () => {
        // Hide tooltip
        tooltip.style('opacity', 0);
      })
      .on('click', (event, d) => {
        // Select point on click
        setSelectedPoint(d);
        event.stopPropagation();
      });
    
    // Add click handler to clear selection when clicking on empty space
    svg.on('click', () => {
      setSelectedPoint(null);
    });
    
    // Add a legend for file types
    const fileTypes = new Set();
    filteredPoints.forEach(point => {
      if (point.metadata && point.metadata.file_type) {
        fileTypes.add(point.metadata.file_type);
      }
    });
    
    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${width - 120}, 20)`);
    
    Array.from(fileTypes).forEach((fileType, i) => {
      const legendItem = legend.append('g')
        .attr('transform', `translate(0, ${i * 20})`);
      
      legendItem.append('rect')
        .attr('width', 10)
        .attr('height', 10)
        .attr('fill', colorScale(fileType));
      
      legendItem.append('text')
        .attr('x', 15)
        .attr('y', 10)
        .attr('font-size', '12px')
        .text(fileType);
    });
  };
  
  // Handle parameter changes
  const handleMethodChange = (selectedOption) => {
    setMethod(selectedOption);
    fetchVisualizationData();
  };
  
  const handlePerplexityChange = (e) => {
    setPerplexity(parseInt(e.target.value));
  };
  
  const handleNNeighborsChange = (e) => {
    setNNeighbors(parseInt(e.target.value));
  };
  
  const handleMinDistChange = (e) => {
    setMinDist(parseFloat(e.target.value));
  };
  
  const handleFilterChange = (e) => {
    setFilterText(e.target.value);
  };
  
  const handleApplyParameters = () => {
    fetchVisualizationData();
  };
  
  return (
    <div className="vector-visualization">
      <h2 className="text-xl font-semibold mb-4">Vector Space Visualization</h2>
      
      {/* Control Panel */}
      <div className="control-panel bg-gray-100 p-4 rounded-lg mb-4">
        <h3 className="text-lg font-medium mb-2">Visualization Controls</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Method Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Dimensionality Reduction Method
            </label>
            <Select
              value={method}
              onChange={handleMethodChange}
              options={methodOptions}
              className="basic-single"
              classNamePrefix="select"
            />
          </div>
          
          {/* Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Filter Documents
            </label>
            <input
              type="text"
              value={filterText}
              onChange={handleFilterChange}
              placeholder="Filter by title, filename, or content"
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
            />
          </div>
          
          {/* Method-specific parameters */}
          {method.value === 'tsne' && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Perplexity: {perplexity}
              </label>
              <input
                type="range"
                min="5"
                max="50"
                value={perplexity}
                onChange={handlePerplexityChange}
                className="w-full"
              />
              <span className="text-xs text-gray-500">
                Higher values consider more global structure
              </span>
            </div>
          )}
          
          {method.value === 'umap' && (
            <>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Number of Neighbors: {nNeighbors}
                </label>
                <input
                  type="range"
                  min="2"
                  max="100"
                  value={nNeighbors}
                  onChange={handleNNeighborsChange}
                  className="w-full"
                />
                <span className="text-xs text-gray-500">
                  Higher values consider more global structure
                </span>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Minimum Distance: {minDist}
                </label>
                <input
                  type="range"
                  min="0.01"
                  max="0.99"
                  step="0.01"
                  value={minDist}
                  onChange={handleMinDistChange}
                  className="w-full"
                />
                <span className="text-xs text-gray-500">
                  Lower values create tighter clusters
                </span>
              </div>
            </>
          )}
          
          {/* Apply Button */}
          <div className="md:col-span-2">
            <button
              onClick={handleApplyParameters}
              className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              Apply Parameters
            </button>
          </div>
        </div>
      </div>
      
      {/* Visualization Area */}
      <div className="visualization-container relative bg-white p-4 rounded-lg shadow" data-testid="visualization-container">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75">
            <div className="text-lg font-medium text-gray-700">Loading visualization...</div>
          </div>
        )}
        
        {error && (
          <div className="text-red-500 mb-4">{error}</div>
        )}
        
        <div className="flex flex-col md:flex-row">
          {/* SVG Container */}
          <div className="flex-grow">
            <svg ref={svgRef} className="w-full h-[600px]"></svg>
            <div 
              ref={tooltipRef} 
              className="tooltip absolute bg-white p-2 rounded shadow-lg border border-gray-200 text-sm pointer-events-none opacity-0 transition-opacity"
              style={{ zIndex: 1000 }}
            ></div>
          </div>
          
          {/* Selected Document Details */}
          {selectedPoint && (
            <div className="md:w-1/3 p-4 bg-gray-50 rounded-lg ml-0 md:ml-4 mt-4 md:mt-0">
              <h3 className="text-lg font-medium mb-2">Document Details</h3>
              
              {selectedPoint.metadata ? (
                <div>
                  <p><strong>Title:</strong> {selectedPoint.metadata.title || 'Untitled'}</p>
                  <p><strong>File Type:</strong> {selectedPoint.metadata.file_type || 'Unknown'}</p>
                  {selectedPoint.metadata.file_size && (
                    <p><strong>Size:</strong> {Math.round(selectedPoint.metadata.file_size / 1024)} KB</p>
                  )}
                  {selectedPoint.metadata.word_count && (
                    <p><strong>Word Count:</strong> {selectedPoint.metadata.word_count}</p>
                  )}
                  {selectedPoint.metadata.upload_timestamp && (
                    <p><strong>Uploaded:</strong> {new Date(selectedPoint.metadata.upload_timestamp).toLocaleString()}</p>
                  )}
                  
                  {selectedPoint.metadata.content_preview && (
                    <div className="mt-2">
                      <p><strong>Preview:</strong></p>
                      <div className="mt-1 p-2 bg-white rounded border border-gray-200 text-sm max-h-40 overflow-y-auto">
                        {selectedPoint.metadata.content_preview}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <p>No metadata available for this document.</p>
              )}
            </div>
          )}
        </div>
        
        {/* Instructions */}
        <div className="mt-4 text-sm text-gray-600">
          <p><strong>Instructions:</strong> Scroll to zoom, drag to pan. Click on a point to view document details.</p>
        </div>
      </div>
    </div>
  );
};

export default VectorVisualization;
