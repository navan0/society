# Next.js + Tailwind + shadcn/ui Boilerplate

A clean starter you can `git init` on top of.

## What you get
- Next.js App Router (TypeScript)
- TailwindCSS with shadcn/ui tokens + dark mode
- Pre-wired `Button` and `Card` components
- Theme toggle using `next-themes`
- ESLint + Prettier

## Quickstart

> Requires **Node.js 18.17+** or **Node 20+**.

```bash
# 1) Install deps (pick one)
pnpm install
# or
npm install
# or
yarn

# 2) Run dev server
pnpm dev
# or: npm run dev

# 3) Build & run production
pnpm build && pnpm start
```

Visit http://localhost:3000

## Customize
- Add more components from shadcn/ui by copying their code into `components/ui/*`.
- Tweak theme tokens in `app/globals.css`.
- Update metadata in `app/layout.tsx`.
