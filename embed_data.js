const fs = require('fs');

// Load JSON data
const diseaseData = JSON.stringify(require('./disease_plants.json'));
const vernacularData = JSON.stringify(require('./vernacular_data.json'));
const plantList = JSON.stringify(require('./medicinal_plant.json'));

// Read directory.html
let html = fs.readFileSync('directory.html', 'utf8');

// Find and replace the entire script block
const scriptStart = html.indexOf('<script>');
const scriptEnd = html.indexOf('</script>') + '</script>'.length;
const oldScript = html.substring(scriptStart, scriptEnd);

const newScript = `<script>
        let diseaseData = ${diseaseData};
        let vernacularData = ${vernacularData};
        let plantList = ${plantList};
        let filteredData = [...diseaseData];

        function getVernacularNames(botanical) {
            const matches = vernacularData.filter(v =>
                v.botanical_equivalent && v.botanical_equivalent.toLowerCase() === botanical.toLowerCase()
            );
            return matches.map(m => m.vernacular_name + " (" + m.language_tribe + ")");
        }

        function processData() {
            const query = searchInput.value.toLowerCase().trim();
            const sortValue = sortSelect.value;

            filteredData = diseaseData.filter(item => {
                const diseaseMatches = item.disease.toLowerCase().includes(query);
                const plantMatches = item.plants.some(p => p.toLowerCase().includes(query));
                const vernacularMatches = vernacularData.some(v =>
                    v.vernacular_name.toLowerCase().includes(query) ||
                    v.botanical_equivalent.toLowerCase().includes(query)
                );
                return diseaseMatches || plantMatches || vernacularMatches;
            });

            filteredData.sort((a, b) => {
                if (sortValue === "az") return a.disease.localeCompare(b.disease);
                if (sortValue === "za") return b.disease.localeCompare(a.disease);
                if (sortValue === "count_desc") return b.plants.length - a.plants.length;
                if (sortValue === "count_asc") return a.plants.length - b.plants.length;
            });

            renderData();
        }

        function renderData() {
            const gallery = document.getElementById("gallery");
            gallery.innerHTML = "";
            document.getElementById("resultsCount").textContent =
                filteredData.length + " result" + (filteredData.length !== 1 ? "s" : "") + " found";

            if (filteredData.length === 0) {
                gallery.innerHTML = '<div class="no-results">No diseases or plants found matching your search.</div>';
                return;
            }

            filteredData.forEach(item => {
                const card = document.createElement("div");
                card.className = "card";

                const title = document.createElement("h2");
                title.className = "disease-title";
                title.textContent = item.disease;

                const list = document.createElement("div");
                list.className = "plant-list";

                item.plants.forEach(plant => {
                    const tag = document.createElement("span");
                    tag.className = "plant-tag";
                    tag.textContent = plant;
                    list.appendChild(tag);
                });

                card.appendChild(title);
                card.appendChild(list);
                gallery.appendChild(card);
            });
        }

        document.getElementById("searchInput").addEventListener("input", processData);
        document.getElementById("sortSelect").addEventListener("change", processData);

        renderData();
    </script>`;

html = html.replace(oldScript, newScript);
fs.writeFileSync('directory.html', html);
console.log('Done! Embedded data directly into directory.html');