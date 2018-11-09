import {Component, OnInit} from '@angular/core';
import {ActivatedRoute, Router} from '@angular/router';
import {Http} from '@angular/http';
import {IP_ADDRESS} from '../data';

@Component({
  selector: 'app-result',
  templateUrl: './result.component.html',
  styleUrls: ['./result.component.css']
})
export class ResultComponent implements OnInit {

  private selectedDrinkId: number;
  private content = '';

  constructor(private activatedRoute: ActivatedRoute,
              private http: Http,
              private router: Router) {
  }

  ngOnInit() {
    this.activatedRoute.params
      .map(params => params['id'])
      .subscribe(id => {
        this.selectedDrinkId = parseInt(id, 0);
        this.setupContent(this.selectedDrinkId);
        this.getCommand();
      });
  }

  getCommand(): void {
    const timerId = setInterval(timer => {
      this.http.get(IP_ADDRESS + '/resultPage').subscribe(data => {
        console.log(data);
        if (data['_body'] === 'Next') {
          this.navigate();
          clearInterval(timerId);
        }
      });
    }, 2500);
  }

  setupContent(id: number): void {
    if (id === 1) {
      this.http.get('assets/result1.txt').subscribe(data => {
        const response: string = data['_body'];
        const paragraph: string[] = response.split('\n');
        let track = 0;
        const timerId = setInterval(timer => {
          if (track < paragraph.length) {
            this.content = paragraph[track];
            track++;
          } else {
            clearInterval(timerId);
          }
        }, 2500);
      });
    } else {
      this.http.get('assets/result2.txt').subscribe(data => {
        const response: string = data['_body'];
        const paragraph: string[] = response.split('\n');
        let track = 0;
        const timerId = setInterval(timer => {
          if (track < paragraph.length) {
            this.content = paragraph[track];
            track++;
          } else {
            this.content = 'A lot of bad things will happen to your body ' +
              'and press next to see how you will look like if you get diabetes';
            clearInterval(timerId);
          }
        }, 2500);
      });
    }
  }

  navigate(): void {
    this.router.navigate(['/horrible_image']);
  }


}
