import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import Select from 'react-select';
import apiService from '../services/api';

const Insights = () => {
  // State for insights data and parameters
  const [insightsData, setInsightsData] = useState(null);
  const [similarPairs, setSimilarPairs] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('clusters'); // 'clusters' or 'similar-pairs'
  
  // Clustering parameters
  const [clusteringMethod, setClusteringMethod] = useState({ value: 'kmeans', label: 'K-Means' });
  const [nClusters, setNClusters] = useState(3);
  const [threshold, setThreshold] = useState(0.7);
  const [maxPairs, setMaxPairs] = useState(10);
  
  // Refs for D3 visualization
  const clusterSvgRef = useRef(null);
  const networkSvgRef = useRef(null);
  
  // Method options for dropdown
  const methodOptions = [
    { value: 'kmeans', label: 'K-Means' },
    { value: 'dbscan', label: 'DBSCAN' },
    { value: 'hierarchical', label: 'Hierarchical' }
  ];
  
  // Fetch insights data
  const fetchInsightsData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Prepare parameters based on selected method
      const params = {
        clustering_method: clusteringMethod.value,
        random_state: 42
      };
      
      // Add method-specific parameters
      if (clusteringMethod.value === 'kmeans') {
        params.n_clusters = nClusters;
      }
      
      const data = await apiService.getInsights(params);
      setInsightsData(data);
    } catch (err) {
      console.error('Error fetching insights data:', err);
      setError('Failed to fetch insights data. Please ensure you have documents uploaded.');
    } finally {
      setLoading(false);
    }
  };
  
  // Fetch similar pairs data
  const fetchSimilarPairs = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const params = {
        threshold: threshold,
        max_pairs: maxPairs
      };
      
      const data = await apiService.getSimilarPairs(params);
      setSimilarPairs(data);
    } catch (err) {
      console.error('Error fetching similar pairs data:', err);
      setError('Failed to fetch similar pairs data. Please ensure you have documents uploaded.');
    } finally {
      setLoading(false);
    }
  };
  
  // Initialize data on component mount
  useEffect(() => {
    if (activeTab === 'clusters') {
      fetchInsightsData();
    } else {
      fetchSimilarPairs();
    }
  }, [activeTab]);
  
  // Update visualizations when data changes
  useEffect(() => {
    if (activeTab === 'clusters' && insightsData) {
      renderClusterVisualization();
    } else if (activeTab === 'similar-pairs' && similarPairs) {
      renderNetworkVisualization();
    }
  }, [insightsData, similarPairs, activeTab]);
  
  // Render cluster visualization
  const renderClusterVisualization = () => {
    if (!clusterSvgRef.current || !insightsData || !insightsData.clusters) return;
    
    // Clear previous visualization
    d3.select(clusterSvgRef.current).selectAll('*').remove();
    
    // Set up SVG dimensions
    const width = 800;
    const height = 600;
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Create SVG element
    const svg = d3.select(clusterSvgRef.current)
      .attr('width', width)
      .attr('height', height);
    
    // Create a group for the visualization
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Create a color scale for clusters
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
    
    // Create a zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.5, 10])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });
    
    // Apply zoom behavior to SVG
    svg.call(zoom);
    
    // Create a force simulation
    const simulation = d3.forceSimulation()
      .force('charge', d3.forceManyBody().strength(-50))
      .force('center', d3.forceCenter(innerWidth / 2, innerHeight / 2))
      .force('collision', d3.forceCollide().radius(10));
    
    // Prepare nodes data
    const nodes = [];
    
    insightsData.clusters.forEach((cluster, clusterIndex) => {
      cluster.documents.forEach(doc => {
        nodes.push({
          id: doc.id,
          cluster: clusterIndex,
          label: doc.metadata && doc.metadata.title ? doc.metadata.title : `Doc ${doc.id}`,
          metadata: doc.metadata
        });
      });
    });
    
    // Add nodes to the simulation
    simulation.nodes(nodes);
    
    // Add nodes
    const node = g.selectAll('.node')
      .data(nodes)
      .enter()
      .append('circle')
      .attr('class', 'node')
      .attr('r', 5)
      .attr('fill', d => colorScale(d.cluster))
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5);
    
    // Add tooltips
    node.append('title')
      .text(d => d.label);
    
    // Update positions on simulation tick
    simulation.on('tick', () => {
      node
        .attr('cx', d => d.x)
        .attr('cy', d => d.y);
    });
    
    // Add cluster labels
    const clusterLabels = g.selectAll('.cluster-label')
      .data(insightsData.clusters)
      .enter()
      .append('g')
      .attr('class', 'cluster-label')
      .attr('transform', (d, i) => {
        // Calculate centroid of cluster
        const clusterNodes = nodes.filter(node => node.cluster === i);
        const x = d3.mean(clusterNodes, node => node.x);
        const y = d3.mean(clusterNodes, node => node.y);
        return `translate(${x},${y})`;
      });
    
    clusterLabels.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '-1.5em')
      .attr('font-weight', 'bold')
      .text((d, i) => `Cluster ${i + 1}: ${d.label || 'Unlabeled'}`);
    
    // Add legend
    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${width - 150}, 20)`);
    
    insightsData.clusters.forEach((cluster, i) => {
      const legendItem = legend.append('g')
        .attr('transform', `translate(0, ${i * 20})`);
      
      legendItem.append('rect')
        .attr('width', 10)
        .attr('height', 10)
        .attr('fill', colorScale(i));
      
      legendItem.append('text')
        .attr('x', 15)
        .attr('y', 10)
        .attr('font-size', '12px')
        .text(`Cluster ${i + 1}: ${cluster.label || 'Unlabeled'}`);
    });
  };
  
  // Render network visualization for similar pairs
  const renderNetworkVisualization = () => {
    if (!networkSvgRef.current || !similarPairs || !similarPairs.similar_pairs) return;
    
    // Clear previous visualization
    d3.select(networkSvgRef.current).selectAll('*').remove();
    
    // Set up SVG dimensions
    const width = 800;
    const height = 600;
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Create SVG element
    const svg = d3.select(networkSvgRef.current)
      .attr('width', width)
      .attr('height', height);
    
    // Create a group for the visualization
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Create a zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.5, 10])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });
    
    // Apply zoom behavior to SVG
    svg.call(zoom);
    
    // Prepare nodes and links data
    const nodesMap = new Map();
    const links = [];
    
    similarPairs.similar_pairs.forEach(pair => {
      // Add nodes if they don't exist
      if (!nodesMap.has(pair.doc1.id)) {
        nodesMap.set(pair.doc1.id, {
          id: pair.doc1.id,
          label: pair.doc1.preview,
        });
      }
      
      if (!nodesMap.has(pair.doc2.id)) {
        nodesMap.set(pair.doc2.id, {
          id: pair.doc2.id,
          label: pair.doc2.preview,
        });
      }
      
      // Add link
      links.push({
        source: pair.doc1.id,
        target: pair.doc2.id,
        similarity: pair.similarity
      });
    });
    
    const nodes = Array.from(nodesMap.values());
    
    // Create a force simulation
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(d => 200 * (1 - d.similarity)))
      .force('charge', d3.forceManyBody().strength(-100))
      .force('center', d3.forceCenter(innerWidth / 2, innerHeight / 2))
      .force('collision', d3.forceCollide().radius(10));
    
    // Create a color scale for similarity
    const colorScale = d3.scaleLinear()
      .domain([threshold, 1])
      .range(['#ccc', '#0066cc']);
    
    // Add links
    const link = g.selectAll('.link')
      .data(links)
      .enter()
      .append('line')
      .attr('class', 'link')
      .attr('stroke', d => colorScale(d.similarity))
      .attr('stroke-width', d => 1 + 4 * (d.similarity - threshold) / (1 - threshold))
      .attr('stroke-opacity', 0.6);
    
    // Add nodes
    const node = g.selectAll('.node')
      .data(nodes)
      .enter()
      .append('circle')
      .attr('class', 'node')
      .attr('r', 7)
      .attr('fill', '#69b3a2')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));
    
    // Add tooltips
    node.append('title')
      .text(d => d.label);
    
    // Add node labels
    const nodeLabels = g.selectAll('.node-label')
      .data(nodes)
      .enter()
      .append('text')
      .attr('class', 'node-label')
      .attr('text-anchor', 'middle')
      .attr('dy', '-0.5em')
      .attr('font-size', '10px')
      .text(d => {
        const label = d.label;
        return label.length > 20 ? label.substring(0, 20) + '...' : label;
      });
    
    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
      
      node
        .attr('cx', d => d.x)
        .attr('cy', d => d.y);
      
      nodeLabels
        .attr('x', d => d.x)
        .attr('y', d => d.y);
    });
    
    // Drag functions
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }
    
    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }
    
    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
    
    // Add legend for similarity
    const legendWidth = 150;
    const legendHeight = 20;
    
    const legendScale = d3.scaleLinear()
      .domain([threshold, 1])
      .range([0, legendWidth]);
    
    const legendAxis = d3.axisBottom(legendScale)
      .ticks(5)
      .tickFormat(d3.format('.2f'));
    
    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${width - legendWidth - 20}, 20)`);
    
    // Create gradient for legend
    const defs = svg.append('defs');
    
    const gradient = defs.append('linearGradient')
      .attr('id', 'similarity-gradient')
      .attr('x1', '0%')
      .attr('y1', '0%')
      .attr('x2', '100%')
      .attr('y2', '0%');
    
    gradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', colorScale(threshold));
    
    gradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', colorScale(1));
    
    // Add legend rectangle
    legend.append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', 'url(#similarity-gradient)');
    
    // Add legend axis
    legend.append('g')
      .attr('transform', `translate(0, ${legendHeight})`)
      .call(legendAxis);
    
    // Add legend title
    legend.append('text')
      .attr('x', 0)
      .attr('y', -5)
      .attr('font-size', '12px')
      .text('Similarity');
  };
  
  // Handle parameter changes
  const handleMethodChange = (selectedOption) => {
    setClusteringMethod(selectedOption);
  };
  
  const handleNClustersChange = (e) => {
    setNClusters(parseInt(e.target.value));
  };
  
  const handleThresholdChange = (e) => {
    setThreshold(parseFloat(e.target.value));
  };
  
  const handleMaxPairsChange = (e) => {
    setMaxPairs(parseInt(e.target.value));
  };
  
  const handleApplyParameters = () => {
    if (activeTab === 'clusters') {
      fetchInsightsData();
    } else {
      fetchSimilarPairs();
    }
  };
  
  return (
    <div className="insights">
      <h2 className="text-xl font-semibold mb-4">Document Insights</h2>
      
      {/* Tabs */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="-mb-px flex space-x-8">
          <button
            className={`${
              activeTab === 'clusters'
                ? 'border-indigo-500 text-indigo-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
            onClick={() => setActiveTab('clusters')}
          >
            Document Clusters
          </button>
          <button
            className={`${
              activeTab === 'similar-pairs'
                ? 'border-indigo-500 text-indigo-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
            onClick={() => setActiveTab('similar-pairs')}
          >
            Similar Document Pairs
          </button>
        </nav>
      </div>
      
      {/* Control Panel */}
      <div className="control-panel bg-gray-100 p-4 rounded-lg mb-4">
        <h3 className="text-lg font-medium mb-2">Insights Controls</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {activeTab === 'clusters' ? (
            <>
              {/* Clustering Method Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Clustering Method
                </label>
                <Select
                  value={clusteringMethod}
                  onChange={handleMethodChange}
                  options={methodOptions}
                  className="basic-single"
                  classNamePrefix="select"
                />
              </div>
              
              {/* Number of Clusters */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Number of Clusters: {nClusters}
                </label>
                <input
                  type="range"
                  min="2"
                  max="10"
                  value={nClusters}
                  onChange={handleNClustersChange}
                  className="w-full"
                  disabled={clusteringMethod.value !== 'kmeans'}
                />
                <span className="text-xs text-gray-500">
                  Only applicable for K-Means clustering
                </span>
              </div>
            </>
          ) : (
            <>
              {/* Similarity Threshold */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Similarity Threshold: {threshold.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="0.99"
                  step="0.01"
                  value={threshold}
                  onChange={handleThresholdChange}
                  className="w-full"
                />
                <span className="text-xs text-gray-500">
                  Higher values show only more similar documents
                </span>
              </div>
              
              {/* Max Pairs */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Maximum Pairs: {maxPairs}
                </label>
                <input
                  type="range"
                  min="5"
                  max="50"
                  value={maxPairs}
                  onChange={handleMaxPairsChange}
                  className="w-full"
                />
                <span className="text-xs text-gray-500">
                  Maximum number of document pairs to display
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
      <div className="visualization-container relative bg-white p-4 rounded-lg shadow" data-testid="insights-container">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75">
            <div className="text-lg font-medium text-gray-700">Loading insights...</div>
          </div>
        )}
        
        {error && (
          <div className="text-red-500 mb-4">{error}</div>
        )}
        
        {activeTab === 'clusters' ? (
          <>
            <svg ref={clusterSvgRef} className="w-full h-[600px]"></svg>
            {insightsData && insightsData.clusters && (
              <div className="mt-4">
                <h3 className="text-lg font-medium mb-2">Cluster Information</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {insightsData.clusters.map((cluster, index) => (
                    <div key={index} className="bg-gray-50 p-3 rounded-lg">
                      <h4 className="font-medium">Cluster {index + 1}: {cluster.label || 'Unlabeled'}</h4>
                      <p>Documents: {cluster.documents.length}</p>
                      <p>Keywords: {cluster.keywords.join(', ')}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          <>
            <svg ref={networkSvgRef} className="w-full h-[600px]"></svg>
            {similarPairs && similarPairs.similar_pairs && (
              <div className="mt-4">
                <h3 className="text-lg font-medium mb-2">Similar Document Pairs</h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Document 1</th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Document 2</th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Similarity</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {similarPairs.similar_pairs.map((pair, index) => (
                        <tr key={index}>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{pair.doc1.preview}</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{pair.doc2.preview}</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{pair.similarity.toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </>
        )}
        
        {/* Instructions */}
        <div className="mt-4 text-sm text-gray-600">
          <p><strong>Instructions:</strong> Scroll to zoom, drag to pan. Hover over nodes to see document details.</p>
        </div>
      </div>
    </div>
  );
};

export default Insights;
