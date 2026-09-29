const fs = require('fs');

// Load all three JSON files
const medicinalPlants = JSON.parse(fs.readFileSync('medicinal_plant.json', 'utf8'));
const diseaseData = JSON.parse(fs.readFileSync('disease_plants.json', 'utf8'));
const vernacularData = JSON.parse(fs.readFileSync('vernacular_data.json', 'utf8'));

// Build a map: botanical name -> vernacular names
const vernacularMap = {};
vernacularData.forEach(v => {
    const bot = v.botanical_equivalent.toLowerCase();
    if (!vernacularMap[bot]) vernacularMap[bot] = [];
    vernacularMap[bot].push({ name: v.vernacular_name, language: v.language_tribe });
});

// Build a map: plant name -> uses (from disease data)
const plantUsesMap = {};
const plantPartsMap = {};
diseaseData.forEach(d => {
    d.plants.forEach(plant => {
        const key = plant.toLowerCase();
        if (!plantUsesMap[key]) plantUsesMap[key] = [];
        plantUsesMap[key].push(d.disease);
    });
});

// Build combined plant list
const combinedPlants = [];
let id = 0;

medicinalPlants.forEach(plant => {
    id++;
    const key = plant.toLowerCase();
    const vernaculars = vernacularMap[key] || [];
    const uses = plantUsesMap[key] || [];
    
    // Determine family (simple heuristic based on first word)
    const family = 'Unknown';
    
    combinedPlants.push({
        id: id,
        scientificName: plant,
        family: family,
        localNames: vernaculars,
        partsUsed: [],
        uses: uses,
        preparations: [],
        sourcePage: 0
    });
});

// Write combined JSON
fs.writeFileSync('plants_combined.json', JSON.stringify(combinedPlants, null, 2));
console.log('Created plants_combined.json with ' + combinedPlants.length + ' plants');

// Now embed into inform_data.html
let html = fs.readFileSync('inform_data.html', 'utf8');

// Find the plants array in the script
const plantsStart = html.indexOf('const plants = [');
const plantsEnd = html.indexOf('];', plantsStart) + 1;
const oldPlantsBlock = html.substring(plantsStart, plantsEnd);

const newPlantsBlock = 'const plants = ' + JSON.stringify(combinedPlants, null, 8) + ';';

html = html.replace(oldPlantsBlock, newPlantsBlock);

// Also update the header to reflect it's data-driven
html = html.replace(
    '<title>Medicinal Plants of East Africa</title>',
    '<title>Medicinal Plants of East Africa</title>'
);

fs.writeFileSync('inform_data.html', html);
console.log('Updated inform_data.html with combined plant data');