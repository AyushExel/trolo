import { useEffect } from 'react';

export default function HomepageFeatures(): JSX.Element {
  useEffect(() => {
    window.location.href = '/intro';
  }, []);
  
  return null;
}
