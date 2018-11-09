import {Component, OnInit, ViewEncapsulation} from '@angular/core';
import {Http, HttpModule} from '@angular/http';
import {IP_ADDRESS} from '../data';
import {Router} from '@angular/router';

@Component({
  selector: 'app-intro',
  templateUrl: './intro.component.html',
  styleUrls: ['./intro.component.css'],
  encapsulation: ViewEncapsulation.None
})
export class IntroComponent implements OnInit {

  constructor(private http: Http, private router: Router) {
  }

  ngOnInit() {
    this.http.get(IP_ADDRESS).subscribe(data => {
      console.log(data);
    });
    this.getCommand();
  }

  getCommand(): void {
    const timerId = setInterval(timer => {
      this.http.get(IP_ADDRESS + '/getText').subscribe(data => {
        console.log(data);
        if (data['_body'] === 'Next') {
          this.router.navigate(['/menu'], 2000);
          clearInterval(timerId);
        }
      });
    }, 2000);
  }
}
