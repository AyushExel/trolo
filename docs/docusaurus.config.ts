import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';

const config: Config = {
  title: 'Trolo',
  tagline: 'A framework for harnessing the power of transformers with YOLO models and other single-shot detectors!',
  favicon: 'img/favicon.ico',
  url: 'https://ayushexel.github.io',
  baseUrl: '/trolo/',
  organizationName: 'ayushexel',
  projectName: 'trolo',
  deploymentBranch: 'gh-pages',
  trailingSlash: true,
  
  themeConfig: {
    navbar: {
      title: 'Trolo',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Documentation',
        },
        {
          href: 'https://github.com/ayushexel/trolo',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {
              label: 'Quickstart',
              to: '/intro',
            },
            {
              label: 'Models',
              to: '/models',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub Issues',
              href: 'https://github.com/ayushexel/trolo/issues',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} trolo. Built with Docusaurus.`,
    },
  },
  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/ayushexel/trolo/tree/main/docs/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      },
    ],
  ],
};

export default config;
