import { render } from '@testing-library/react';

import VoiceChatbot from './VoiceChatbot';

describe('VoiceChatbot', () => {
  it('should render successfully', () => {
    const { baseElement } = render(<VoiceChatbot />);
    expect(baseElement).toBeTruthy();
  });
});
