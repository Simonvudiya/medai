const fs = require('fs');

// Load JSON data
const diseaseData = JSON.stringify(require('./disease_plants.json'));
const vernacularData = JSON.stringify(require('./vernacular_data.json'));
const plantList = JSON.stringify(require('./medicinal_plant.json'));

// Read directory.html template
let html = fs.readFileSync('directory.html', 'utf8');

// Replace placeholders with actual data
html = html.replace('[DISEASE_DATA_PLACEHOLDER]', diseaseData);
html = html.replace('[VERNACULAR_DATA_PLACEHOLDER]', vernacularData);
html = html.replace('[PLANT_LIST_PLACEHOLDER]', plantList);

// Add tab-bar styles to the HTML
const tabStyles = `
        /* Tab bar */
        .tab-bar {
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        .tab-btn {
            padding: 10px 20px;
            background-color: #e0e0e0;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
        }

        .tab-btn:hover {
            background-color: #d0d0d0;
        }

        .tab-btn.active {
            background-color: var(--accent-color);
            color: white;
        }
`;

// Insert tab styles before </style> or in head
html = html.replace('</head>', tabStyles + '</head>');

fs.writeFileSync('directory.html', html);
console.log('Done! Embedded all data and added tab styles.');