# Public site deployment

This folder is self-contained. Upload the whole `public_site/` directory to any static hosting service.

Recommended options:

1. GitHub Pages
   - Create a GitHub repository.
   - Upload all files inside `public_site/`.
   - Enable Pages from the repository settings.

2. Netlify or Cloudflare Pages
   - Drag and drop this `public_site/` folder in the dashboard.
   - No build command is required.

3. Any static server
   - Serve `public_site/` as the web root.
   - Open `/index.html`.

The site does not need a backend. All chart data is in `kline_data.js`; downloadable CSV files are in `data/`.
