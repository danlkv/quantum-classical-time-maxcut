import numpy as np
import matplotlib.pyplot as plt


class MiplogData:
    """ 
    This class holds information about running gurobi solver 
    on a test problem.
    """
    def __init__(self, miplog):
        """
        Args:
            miplog (np.array): (:,3)-shaped array with times,
                upper bound cost and lower bound cost
        """
        self.miplog = miplog

    def plot(self):
        """ Plot bounds data with respect to time """
        mipd = np.array(self.miplog).T

        start = 2
        opt = mipd[2][-1]
        cost_lb = mipd[2][-1]
        plt.plot(mipd[0][start:], mipd[1][start:]/opt, '*-')
        #plt.plot(mipd[0][start:], mipd[2][start:]/(opt), '*-', label='AR')
        plt.plot(mipd[0][start:], mipd[2][start:]/cost_lb, '*-', label='AR UB')
        plt.plot(mipd[0][start:], mipd[2][start:]/mipd[1][start:], '-', label='AR LB')
        plt.legend()
        plt.xscale('log')
        plt.hlines(1, min(mipd[0][start:]), max(mipd[0]))

    # -- Getting time to achieve different AR bounds
    # The AR is defined as cost(sol)/max_sol(cost).
    # If the problem is hard, we can't know what is max_sol
    # The following functions use upper and lower bounds on max_sol
    # to give time to reach lower and upper bounds on AR respectively

    def get_bounds_at(self, ts):
        times, bounds, costs = self.get_history_data()
        ix = np.where((ts - times)>=0)[0]
        if len(ix) == 0:
            return None, None
        ix = max(ix)
        return bounds[ix], costs[ix]

    def get_history_data(self):
        """
        Returns: times, bounds, costs
        """
        return np.array(self.miplog)[1:].T

    def get_time_for_AR_lower(self, AR):
        """ 
        Args:
            AR (float): which AR bound to look for
        Returns:
            time, AR
        """
        times, bounds, costs = self.get_history_data()
        AR_bound = costs/bounds
        # get the index at which AR bound is higher than requested
        ix = np.where((AR_bound-AR)>=0)[0]
        if len(ix)==0: return None, None
        else: ix = min(ix)
        return times[ix], AR_bound[ix]

    def get_time_for_AR_upper(self, AR):
        """
        Returns time during at least which upper bound is lower than ``AR``
        Args:
            AR (float): which AR bound to look for
        Returns:
            time, AR
        """
        times, bounds, costs = self.get_history_data()
        AR_bound = costs/costs[-1]
        # get the index at which AR bound is higher than requested
        ix = np.where((AR_bound-AR)<=0)[0]
        if len(ix)==0: return None, None
        else: ix = max(ix)
        return times[ix], AR_bound[ix]
    
    # --
    

    @classmethod
    def from_start_and_diffs(cls, starts, diffs):
        timecurr, boundcurr, costcurr = list(starts).copy()
        #print('diffs' ,diffs)

        mipd = [[0, 1e100, 0], [timecurr, boundcurr, costcurr]]
        for type, dt, dval in diffs:
            if type==1:
                dval = -dval
                boundcurr += dval
            else:
                costcurr += dval
            timecurr += dt
            mipd.append([timecurr, boundcurr, costcurr])
        return cls(mipd)


    def get_starting_point(self):
        timestart, currbound, currcost = self.miplog[1]
        return timestart, currbound, currcost

    def get_diff_data(self):
        timestart, currbound, currcost = self.get_starting_point()
        diff_data = []
        for time, bound, cost  in self.miplog[2:]:
            dt = time - timestart
            if bound!=currbound:
                # bound is only decreasing
                dbound = currbound - bound
                #print('dbound', dbound, currbound, bound)
                diff_data.append((1, dt, dbound))
                currbound = bound
                timestart = time
                dt = 1
            if cost!=currcost:
                # cost is only increasing
                dcost = cost - currcost
                #print('dcost', dcost, currcost, cost)
                diff_data.append((2, dt, dcost))
                currcost = cost
                timestart = time
        return diff_data

    # -- Serialization to binary
    # The following methods are used to convert starting points+diffs format
    # to a binary format. This conversion is lossy, with numbers rounded to closest int

    @classmethod
    def from_file(cls, file):
        with open(file, 'rb') as f:
            data = f.read()
        return cls.from_bytes(data)

    @classmethod
    def from_bytes(cls, data):
        start1 = data[:3]
        start2 = data[3:3*2]
        start3 = data[3*2:3*3]
        t0, v0 = cls._bytes_to_starting(start1)
        if t0 not in [1, 2, 3]:
            raise ValueError(f'Type {t0} is not supported')
        t1, v1 = cls._bytes_to_starting(start2)
        if t1 not in [1, 2, 3]:
            raise ValueError(f'Type {t1} is not supported')
        t2, v2 = cls._bytes_to_starting(start3)
        if t2 not in [1, 2, 3]:
            raise ValueError(f'Type {t2} is not supported')
        starts = [0]*3
        type_2_idx = {1:1, 2:2, 3:0}
        starts[type_2_idx[t1]] = v1
        starts[type_2_idx[t2]] = v2
        starts[type_2_idx[t0]] = v0
        #print('starts', starts)

        diffs = cls.diffs_from_bytes(data[3*3:])
        #print('diffs' ,diffs)
        return cls.from_start_and_diffs(starts, diffs)

    def write_bytes(self, file):
        with open(file, 'wb') as f:
            f.write(self.get_bytes())

    @classmethod
    def diffs_from_bytes(cls, data):
        packet_len = 3
        assert len(data)%packet_len == 0 
        packets_cnt = len(data)//3
        bytes = iter(data)
        packets = []
        for i in range(packets_cnt):
            pack = [ ]
            for i in range(packet_len):
                pack.append(next(bytes))
            upd = cls._bytes_to_diff(pack)
            packets.append(upd)

        return packets

    def get_bytes(self):
        bytes = b''
        sttime, stbound, stcost = self.get_starting_point()
        bytes  += self._get_starting_bytes(type=3, value=sttime)
        bytes  += self._get_starting_bytes(type=1, value=stbound)
        bytes  += self._get_starting_bytes(type=2, value=stcost)
        packet_data = self.get_diff_data()
        for diff in packet_data:
            bytes += self._diff_to_bytes(*diff)
        return bytes

    @staticmethod
    def _bytes_to_diff(bytes):
        num = int(bytes[0])*256**2 + int(bytes[1])*256 + int(bytes[2])
        type = num%16 + 1
        time = (num//16)%1024
        value = (num//16//1024)%1024
        return type, time, value+1

    @staticmethod
    def _bytes_to_starting(bytes):
        num = int(bytes[0])*256**2 + int(bytes[1])*256 + int(bytes[2])
        type = num%16 + 1
        value = (num//16)%1024**2
        return type, value

    def _get_starting_bytes(self, type, value):
        """
        value | time | type
        """
        packet = 0
        # type can be from 1 to 16
        if type>16 or type<1:
            raise ValueError('type has to be from 1 to 16')
        packet += int(type-1)
        if value>2**20 or value<0:
            raise ValueError(f'vlaue has to be from 0 to 2^20, but got {value}')
        packet += int(value)*16
        return bytes([(packet//256//256)%256, (packet//256)%256, packet%256])

    
    def _diff_to_bytes(self, type, time, value):
        """
        value | time | type
        """
        packet = 0
        # type can be from 1 to 16
        if type>16 or type<1:
            raise ValueError('type has to be from 1 to 16')
        packet += int(type-1)
        if time>1023 or time<0:
            raise ValueError('time has to be from 0 to 1023')
        # time can be from 0 to 1023
        packet += round(time)*2**4
        if value>1024 or value<1:
            raise ValueError('value has to be from 1 to 1024')
        # valchange can be from 1 to 1024
        packet += round(value-1)*2**4*2**10

        # 10+10+4 total 3 bytes
        #print('bin', bin(packet), f'(length={len(bin(packet))-2})')
        #print('hex', hex(packet))
        return bytes([(packet//256//256)%256, (packet//256)%256, packet%256])




test_data = [[2.384185791015625e-07, 1e+100, -0.0],
 [1.4091267585754395, 7031.0, 5997.0],
 [6.190175294876099, 7031.0, 6056.0],
 [7.662626504898071, 6986.0, 6222.0],
 [15.388627767562866, 6982.0, 6230.0],
 [17.94478702545166, 6967.0, 6383.0],
 [25.34035301208496, 6967.0, 6403.0],
 [72.4750623703003, 6967.0, 6403.0],
 [74.94901704788208, 6967.0, 6403.0],
 [77.25160074234009, 6965.0, 6403.0],
 [78.44797253608704, 6965.0, 6403.0],
 [80.06166100502014, 6965.0, 6403.0],
 [81.2233259677887, 6965.0, 6403.0],
 [82.76489281654358, 6965.0, 6403.0],
 [85.40263867378235, 6965.0, 6403.0],
 [88.77241039276123, 6965.0, 6403.0],
 [93.73358917236328, 6965.0, 6403.0],
 [101.38260245323181, 6965.0, 6403.0],
 [110.38635110855103, 6965.0, 6403.0],
 [119.51304125785828, 6965.0, 6403.0],
 [129.11753463745117, 6965.0, 6403.0],
 [138.81520318984985, 6965.0, 6403.0],
 [147.75759196281433, 6965.0, 6403.0],
 [156.1373565196991, 6965.0, 6406.0],
 [165.70949506759644, 6965.0, 6406.0],
 [177.10741758346558, 6965.0, 6406.0],
 [188.263601064682, 6965.0, 6406.0],
 [247.9679811000824, 6965.0, 6406.0],
 [261.40310978889465, 6965.0, 6406.0],
 [276.9358479976654, 6965.0, 6406.0],
 [291.55994939804077, 6965.0, 6406.0],
 [306.5596799850464, 6965.0, 6406.0],
 [320.32182359695435, 6965.0, 6406.0],
 [331.9800500869751, 6965.0, 6407.0],
 [488.00892901420593, 6965.0, 6407.0],
 [496.26656794548035, 6965.0, 6408.0],
 [505.78080201148987, 6965.0, 6408.0],
 [512.6630666255951, 6965.0, 6410.0],
 [520.3788964748383, 6965.0, 6413.0],
 [526.9174110889435, 6965.0, 6416.0],
 [534.6312460899353, 6965.0, 6419.0],
 [541.7277989387512, 6965.0, 6428.0],
 [548.3258099555969, 6965.0, 6446.0],
 [555.413773059845, 6965.0, 6452.0],
 [562.3991062641144, 6965.0, 6452.0],
 [569.2310676574707, 6965.0, 6456.0],
 [576.1856963634491, 6965.0, 6456.0],
 [583.7974467277527, 6965.0, 6460.0],
 [591.7745225429535, 6965.0, 6460.0],
 [599.8289680480957, 6965.0, 6464.0],
 [610.1037223339081, 6965.0, 6464.0],
 [620.247725725174, 6965.0, 6465.0],
 [629.5615994930267, 6965.0, 6465.0],
 [636.887289762497, 6965.0, 6466.0],
 [644.7588560581207, 6965.0, 6466.0]]


def test_miplog_helper():
    mh = MiplogData(test_data)
    packets = mh.get_diff_data()
    bytes = mh.get_bytes()
    assert len(packets)*3 + 3*3 == len(bytes)

    mh2 = MiplogData.from_bytes(bytes)
    print('read data', mh2.get_diff_data())
    rounded = [(int(t), round(ti), round(v)) for t, ti, v in mh.get_diff_data()]
    print('rounded', rounded)
    assert rounded == mh2.get_diff_data()


    mh = MiplogData([[0, 1e100, 0], [1, 200, 300]])
    packets = mh.get_diff_data()
    bytes = mh.get_bytes()
    assert len(packets)*3 + 3*3 == len(bytes)
    assert len(packets)==0
    mh = MiplogData([[0, 1e100, 0], [1, 200, 300], [2, 0, 100]])
    packets = mh.get_diff_data()
    assert packets[0] == (1, 1, 200)
