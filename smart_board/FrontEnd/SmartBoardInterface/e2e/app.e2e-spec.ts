import { SmartBoardInterfacePage } from './app.po';

describe('smart-board-interface App', function() {
  let page: SmartBoardInterfacePage;

  beforeEach(() => {
    page = new SmartBoardInterfacePage();
  });

  it('should display message saying app works', () => {
    page.navigateTo();
    expect(page.getParagraphText()).toEqual('app works!');
  });
});
