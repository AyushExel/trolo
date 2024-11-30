const sidebars = {
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'intro',
      label: 'Quickstart',
    },
    {
      type: 'category',
      label: 'Models',
      items: ['models/index', 'models/d-fine'],
    },
    {
      type: 'category',
      label: 'Features',
      items: [
        'features/inference',
        'features/training',
      ],
    },
  ],
};

export default sidebars; 