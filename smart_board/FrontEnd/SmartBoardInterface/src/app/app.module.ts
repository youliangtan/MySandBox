import {BrowserModule} from '@angular/platform-browser';
import {NgModule} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {HttpModule} from '@angular/http';

import {AppComponent} from './app.component';
import {IntroComponent} from './intro/intro.component';
import {MenuComponent} from './menu/menu.component';
import {RouterModule} from '@angular/router';
import {DrinksLoadingComponent} from './drinks-loading/drinks-loading.component';
import {ResultComponent} from './result/result.component';
import {HorribleImageComponent} from './horrible-image/horrible-image.component';
import {MoreInfoComponent} from './more-info/more-info.component';

@NgModule({
  declarations: [
    AppComponent,
    IntroComponent,
    MenuComponent,
    DrinksLoadingComponent,
    ResultComponent,
    HorribleImageComponent,
    MoreInfoComponent
  ],
  imports: [
    BrowserModule,
    FormsModule,
    HttpModule,
    RouterModule,
    RouterModule.forRoot([
      {path: '', redirectTo: '/intro', pathMatch: 'full'},
      {path: 'intro', component: IntroComponent},
      {path: 'menu', component: MenuComponent},
      {path: 'loading/:id', component: DrinksLoadingComponent},
      {path: 'result/:id', component: ResultComponent},
      {path: 'horrible_image', component: HorribleImageComponent},
      {path: 'more_info', component: MoreInfoComponent},
    ])
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule {
}
