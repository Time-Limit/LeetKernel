#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_59_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<59> mm_instantiator;

public:
  UnalignedM_59_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_59_MMInstantiatorWrapper__;

}  // namespace LLMMM
