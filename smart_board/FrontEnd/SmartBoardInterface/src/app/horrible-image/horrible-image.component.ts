import {Component, OnInit} from '@angular/core';
import {IP_ADDRESS} from '../data';
import {Http} from '@angular/http';
import {Router} from '@angular/router';

@Component({
  selector: 'app-horrible-image',
  templateUrl: './horrible-image.component.html',
  styleUrls: ['./horrible-image.component.css']
})
export class HorribleImageComponent implements OnInit {

  constructor(private http: Http, private router: Router) {
  }

  ngOnInit() {
    this.getCommand();
  }

  getCommand(): void {
    const timerId = setInterval(timer => {
      this.http.get(IP_ADDRESS + '/horribleImage').subscribe(data => {
        console.log(data);
        if (data['_body'] === 'Next') {
          this.navigate();
          clearInterval(timerId);
        }
      });
    }, 2000);
  }

  navigate(): void {
    this.router.navigate(['/more_info']);
  }


}
